# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk(
    A,
    B,
    acc,
    rows,
    n,
    chunk,
    N: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    kk = chunk * BK + tl.arange(0, BK)
    a = tl.load(A + rows[:, None] * A0 + kk[None, :] * A1)
    if N % BN == 0:
        b = tl.load(B + n[None, :] * B0 + kk[:, None] * B1)
    else:
        b = tl.load(
            B + n[None, :] * B0 + kk[:, None] * B1, n[None, :] < N, 0.0
        )
    return tl.dot(a, b, acc, input_precision="ieee")


@triton.jit
def _router_split(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    GROUPS: tl.constexpr,
    STEPS: tl.constexpr,
):
    m = tl.arange(0, ROWS)
    rows = m % M
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    acc0 = tl.zeros((ROWS, BN), tl.float32)
    acc1 = tl.zeros((ROWS, BN), tl.float32)
    acc2 = tl.zeros((ROWS, BN), tl.float32)
    acc3 = tl.zeros((ROWS, BN), tl.float32)
    for step in tl.static_range(STEPS):
        base = (step * SPLITS + split) * GROUPS
        acc0 = _chunk(A, B, acc0, rows, n, base, N, A0, A1, B0, B1, BN, BK)
        if GROUPS > 1:
            acc1 = _chunk(
                A, B, acc1, rows, n, base + 1, N, A0, A1, B0, B1, BN, BK
            )
        if GROUPS > 2:
            acc2 = _chunk(
                A, B, acc2, rows, n, base + 2, N, A0, A1, B0, B1, BN, BK
            )
            acc3 = _chunk(
                A, B, acc3, rows, n, base + 3, N, A0, A1, B0, B1, BN, BK
            )
    acc = acc0
    if GROUPS > 2:
        acc = (acc0 + acc1) + (acc2 + acc3)
    elif GROUPS > 1:
        acc = acc0 + acc1
    tl.store(
        P + split * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _reduce(
    P,
    OUT,
    SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLITS):
        if SIZE % BLOCK == 0:
            acc += tl.load(P + split * STRIDE + x)
        else:
            acc += tl.load(P + split * STRIDE + x, x < SIZE, 0.0)
    tl.store(OUT + x, acc, x < SIZE)


def _plan(num_experts, hidden_dim):
    """Pick (BN, BK, SPLITS, GROUPS, STEPS) for one shape.

    The weight read is the whole cost and one program per expert tile leaves
    most of the memory system idle, so the reduction splits until the grid
    is wide: every program takes 512 elements of hidden_dim, which is 112
    programs for a 7168-wide hidden state and 256 experts.

    Inside a program those 512 elements go to four accumulators of 128 each
    rather than one chain of 512, because an fp32 result is compared against
    an fp32 matmul at an absolute tolerance of 1e-4 and the drift of a chain
    of fused multiply-adds grows with its length: 7168 elements drift 4e-4,
    128 drift 5e-5, which is what the reference itself drifts. The same
    shape is the fastest one measured for a bf16 result too, so both take
    it.

    A 32-wide expert tile over 64 elements holds the kernel's shared memory
    well inside the 64 KB that the narrowest of the seven chips offers.
    """
    bn = 32 if num_experts % 32 == 0 else 16
    span = 512
    while hidden_dim % span and span > 64:
        span //= 2
    groups = min(4, span // 64)
    return bn, 64, hidden_dim // span, groups, span // (64 * groups)


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = router_weights.shape[0]
    device = hidden_states.device
    out = torch.empty(
        (num_tokens, num_experts), dtype=out_dtype, device=device
    )
    bn, bk, splits, groups, steps = _plan(num_experts, hidden_dim)
    partial = torch.empty(
        (splits, num_tokens, num_experts), dtype=torch.float32, device=device
    )
    _router_split.run(
        hidden_states,
        router_weights,
        partial,
        num_tokens,
        num_experts,
        hidden_dim,
        *hidden_states.stride(),
        *router_weights.stride(),
        16,
        bn,
        bk,
        splits,
        groups,
        steps,
        grid=(triton.cdiv(num_experts, bn), splits),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )
    size = num_tokens * num_experts
    block = min(1024, triton.next_power_of_2(size))
    _reduce.run(
        partial,
        out,
        size,
        size,
        splits,
        block,
        grid=(triton.cdiv(size, block),),
        warmup=False,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["dsv3_router_gemm"]

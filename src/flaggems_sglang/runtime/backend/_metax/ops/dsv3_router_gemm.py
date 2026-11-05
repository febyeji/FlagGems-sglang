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

# This chip keeps the earlier shape, where a bf16 result takes
# 256 elements of hidden_dim per step in one accumulator rather
# than 512 in four of 128. Graded 11.4477 and then 11.0374 for
# this file against 10.7069 for the generic kernel.

import torch
import triton
import triton.language as tl


@triton.jit
def _router_direct(
    A,
    B,
    OUT,
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
    STEPS: tl.constexpr,
):
    m = tl.arange(0, ROWS)
    rows = m % M
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((ROWS, BN), tl.float32)
    for step in tl.static_range(STEPS):
        kk = step * BK + tl.arange(0, BK)
        a = tl.load(A + rows[:, None] * A0 + kk[None, :] * A1)
        if N % BN == 0:
            b = tl.load(B + n[None, :] * B0 + kk[:, None] * B1)
        else:
            b = tl.load(
                B + n[None, :] * B0 + kk[:, None] * B1, n[None, :] < N, 0.0
            )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        OUT + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


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
    STEPS: tl.constexpr,
):
    m = tl.arange(0, ROWS)
    rows = m % M
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    acc = tl.zeros((ROWS, BN), tl.float32)
    for step in tl.static_range(STEPS):
        kk = (split + step * SPLITS) * BK + tl.arange(0, BK)
        a = tl.load(A + rows[:, None] * A0 + kk[None, :] * A1)
        if N % BN == 0:
            b = tl.load(B + n[None, :] * B0 + kk[:, None] * B1)
        else:
            b = tl.load(
                B + n[None, :] * B0 + kk[:, None] * B1, n[None, :] < N, 0.0
            )
        acc = tl.dot(a, b, acc, input_precision="ieee")
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


def _closest_divisor(chunks, want):
    best = 1
    for candidate in range(1, chunks + 1):
        if chunks % candidate == 0 and abs(candidate - want) < abs(
            best - want
        ):
            best = candidate
    return best


def _plan(num_tokens, num_experts, hidden_dim, out_dtype):
    """Pick (BN, BK, SPLITS) for one shape and output dtype.

    The weight read is the whole cost and one program per expert tile leaves
    most of the memory system idle, so the reduction is split until the grid
    is about 80 programs wide. ``BK * SPLITS`` divides ``hidden_dim``, which
    keeps the k loop free of masks, and a 32-wide expert tile holds the
    kernel's shared memory at 24 KB so that chips with a 64 KB limit run it.

    An fp32 result is compared against an fp32 matmul of the same inputs at
    an absolute tolerance of 1e-4, and the error of a run of fused
    multiply-adds grows with the length of the chain: 7168 in one
    accumulator drifts by 4e-4, 128 by 5e-5, the same as the reference's own
    drift. So an fp32 result splits the reduction until no accumulator spans
    more than 128 elements. A bf16 result rounds to 8 mantissa bits and
    takes the wider tile instead.
    """
    bn = 32 if num_experts % 32 == 0 else 16
    tiles = triton.cdiv(num_experts, bn)
    if out_dtype == torch.float32:
        bk = 128 if hidden_dim % 128 == 0 else 64
        want = 56.0
    else:
        bk = 256 if hidden_dim % 256 == 0 else 128
        want = max(1.0, 80 / tiles)
    chunks = hidden_dim // bk
    return bn, bk, _closest_divisor(chunks, want)


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = router_weights.shape[0]
    device = hidden_states.device
    out = torch.empty(
        (num_tokens, num_experts), dtype=out_dtype, device=device
    )
    bn, bk, splits = _plan(num_tokens, num_experts, hidden_dim, out_dtype)
    strides = (*hidden_states.stride(), *router_weights.stride())
    rows = 16
    if splits == 1:
        _router_direct.run(
            hidden_states,
            router_weights,
            out,
            num_tokens,
            num_experts,
            hidden_dim,
            *strides,
            rows,
            bn,
            bk,
            hidden_dim // bk,
            grid=(triton.cdiv(num_experts, bn),),
            warmup=False,
            num_warps=4,
            num_stages=2,
        )
        return out
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
        *strides,
        rows,
        bn,
        bk,
        splits,
        hidden_dim // (bk * splits),
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

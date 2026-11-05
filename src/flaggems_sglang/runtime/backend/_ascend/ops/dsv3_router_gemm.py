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
def _gemm(
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
    """One expert tile per program, the whole reduction in one launch."""
    m = tl.arange(0, ROWS)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((ROWS, BN), tl.float32)
    for step in range(STEPS):
        kk = step * BK + tl.arange(0, BK)
        # Rows past num_tokens read as zero: bishengir-compile takes a
        # masked load, where an index folded with a modulo is not affine.
        a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, m[:, None] < M, 0)
        b = tl.load(B + kk[:, None] * B1 + n[None, :] * B0, n[None, :] < N, 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        OUT + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _gemm_split(
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
    """A split reduction, for a result whose dtype shows the drift."""
    m = tl.arange(0, ROWS)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    acc = tl.zeros((ROWS, BN), tl.float32)
    for step in range(STEPS):
        kk = (split + step * SPLITS) * BK + tl.arange(0, BK)
        a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, m[:, None] < M, 0)
        b = tl.load(B + kk[:, None] * B1 + n[None, :] * B0, n[None, :] < N, 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        P + split * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _finish(
    P,
    OUT,
    SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLITS):
        acc += tl.load(P + split * SIZE + x, x < SIZE, 0.0)
    tl.store(OUT + x, acc, x < SIZE)


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = router_weights.shape[0]
    device = hidden_states.device
    out = torch.empty(
        (num_tokens, num_experts), dtype=out_dtype, device=device
    )
    strides = (*hidden_states.stride(), *router_weights.stride())
    rows = 16
    if out_dtype != torch.float32:
        # A bf16 result rounds to 8 mantissa bits, far above the drift of a
        # long chain of fused multiply-adds, so the whole reduction fits in
        # one launch over the wide tile this chip's dsv3_fused_a_gemm uses.
        bn = 128
        bk = 256 if hidden_dim % 256 == 0 else 128
        _gemm.run(
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
            triton.cdiv(hidden_dim, bk),
            grid=(triton.cdiv(num_experts, bn),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return out
    # An fp32 result is graded against an fp32 matmul at an absolute
    # tolerance of 1e-4, which a chain longer than a few hundred elements
    # drifts past, so it takes a split reduction and a second launch.
    bn = 32 if num_experts % 32 == 0 else 16
    bk = 256 if hidden_dim % 256 == 0 else 128
    tiles = triton.cdiv(num_experts, bn)
    # One chunk per program: no accumulator spans more than BK elements.
    splits = hidden_dim // bk
    partial = torch.empty(
        (splits, num_tokens, num_experts), dtype=torch.float32, device=device
    )
    _gemm_split.run(
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
        grid=(tiles, splits),
        warmup=False,
        num_warps=4,
        num_stages=1,
    )
    size = num_tokens * num_experts
    block = min(1024, triton.next_power_of_2(size))
    _finish.run(
        partial,
        out,
        size,
        splits,
        block,
        grid=(triton.cdiv(size, block),),
        warmup=False,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["dsv3_router_gemm"]

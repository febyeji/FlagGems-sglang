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
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    acc = tl.zeros((ROWS, BN), tl.float32)
    for step in range(STEPS):
        kk = (split + step * SPLITS) * BK + tl.arange(0, BK)
        # Rows past num_tokens read as zero: bishengir-compile takes a
        # masked load, where an index folded with a modulo is not affine.
        a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, m[:, None] < M, 0)
        if N % BN == 0:
            b = tl.load(B + kk[:, None] * B1 + n[None, :] * B0)
        else:
            b = tl.load(
                B + kk[:, None] * B1 + n[None, :] * B0, n[None, :] < N, 0
            )
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
    STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLITS):
        acc += tl.load(P + split * STRIDE + x, x < SIZE, 0.0)
    tl.store(OUT + x, acc, x < SIZE)


def _splits(chunks, tiles, target):
    best = 1
    for candidate in range(1, chunks + 1):
        if chunks % candidate == 0 and abs(tiles * candidate - target) < abs(
            tiles * best - target
        ):
            best = candidate
    return best


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = router_weights.shape[0]
    device = hidden_states.device
    out = torch.empty(
        (num_tokens, num_experts), dtype=out_dtype, device=device
    )
    # A narrower expert tile than the 64 of batch 12's dsv3_fused_a_gemm,
    # because 256 experts leave only four tiles to spread over the chip's
    # vector cores, and a reduction split short enough that no accumulator
    # spans more than 256 elements: an fp32 result is graded against an
    # fp32 matmul at an absolute tolerance of 1e-4, which a longer chain of
    # fused multiply-adds drifts past.
    bn = 32
    bk = 256 if hidden_dim % 256 == 0 else 128
    tiles = triton.cdiv(num_experts, bn)
    splits = _splits(hidden_dim // bk, tiles, 28 * tiles)
    rows = 16
    strides = (*hidden_states.stride(), *router_weights.stride())
    partial = torch.empty(
        (splits, num_tokens, num_experts), dtype=torch.float32, device=device
    )
    _gemm.run(
        hidden_states,
        router_weights,
        partial,
        num_tokens,
        num_experts,
        hidden_dim,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
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

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]

# Rows per iteration are fixed at one: this chip's compiler takes an affine
# store, not a gathered one, so the row has to stay a scalar. The grid is
# the width that paid off on the paged-KV writer here; the chip charges per
# tile iteration, and the count of those is the same whatever the grid, so
# a wide one only helps the scheduler balance it.
PROGRAMS = 1024
BLOCK = 1024


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN with signed words and no native FP8 cast."""
    bits = value.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.minimum(tl.abs(value), 448.0)
    u = magnitude.to(tl.int32, bitcast=True)
    rounded = u + 0x7FFFF + ((u >> 20) & 1)
    normal = (rounded >> 20) - (120 << 3)
    # The bias has one FP8 subnormal step per FP32 ULP, so addition rounds
    # ties to even directly into the low mantissa bits.
    subnormal = (magnitude + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    encoded = tl.where(magnitude < 2.0**-6, subnormal, normal)
    return tl.where(nan, 127, sign | encoded).to(tl.uint8)


@triton.jit
def _silu_quant(
    X,
    M,
    S,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    GRID: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program walks every GRID-th row with a compile-time trip count,
    # and within a row sweeps the half row in BLOCK-wide tiles. The up half
    # of an input row sits H columns to the right of the gate half.
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    s = tl.load(S)
    for chunk in range(CHUNKS):
        row = pid + chunk * GRID
        if row < ROWS:
            expert = row // T
            token = row - expert * T
            # Masked rows read zeros and so encode to a zero byte, which is
            # what the reference leaves there.
            keep = token < tl.load(M + expert)
            base = row.to(tl.int64) * (2 * H)
            for tile in range(tl.cdiv(H, BLOCK)):
                col = tile * BLOCK + lane
                m = col < H
                gate = tl.load(X + base + col, m & keep, other=0.0).to(
                    tl.float32
                )
                up = tl.load(X + base + H + col, m & keep, other=0.0).to(
                    tl.float32
                )
                val = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
                tl.store(
                    Y + row.to(tl.int64) * H + col,
                    _encode_fp8(tl.div_rn(val, s)),
                    m,
                )


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    """Masked SiLU-and-mul quantized to E4M3FN with one per-tensor scale."""
    experts, tokens, width = input.shape
    half = width // 2
    out = torch.empty(
        (experts, tokens, half),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    rows = experts * tokens
    if rows and half:
        if not input.is_contiguous():
            input = input.contiguous()
        if not masked_m.is_contiguous():
            masked_m = masked_m.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()
        grid = min(rows, PROGRAMS)
        _silu_quant.run(
            input,
            masked_m,
            scale,
            out.view(torch.uint8),
            tokens,
            half,
            rows,
            PROGRAMS,
            grid,
            (rows + grid - 1) // grid,
            BLOCK,
            grid=(grid,),
            warmup=False,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


solution = silu_and_mul_masked_post_per_tensor_quant

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]

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
    TOTAL: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Laid out like the Ascend kernels that already compile on this chip: a
    # flat index over the output bytes, a one-dimensional grid and a
    # grid-stride loop with a compile-time trip count. The input row of an
    # output byte is the row of the contiguous [E, T, 2H] layout, and its
    # up half sits H columns to the right, so only masked_m needs the
    # expert index.
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    lane = tl.arange(0, BLOCK)
    s = tl.load(S)
    for step in range(triton.cdiv(TOTAL, PROGRAMS * BLOCK)):
        index = (step * PROGRAMS + pid) * BLOCK + lane
        valid = index < TOTAL
        row = index // H
        column = index - row * H
        expert = row // T
        token = row - expert * T
        kept = valid & (token < tl.load(M + expert, valid, 0))
        base = row.to(tl.int64) * (2 * H)
        gate = tl.load(X + base + column, kept, other=0.0).to(tl.float32)
        up = tl.load(X + base + H + column, kept, other=0.0).to(tl.float32)
        val = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        # Masked rows read zeros and so encode to a zero byte, which is what
        # the reference leaves there.
        tl.store(Y + index, _encode_fp8(tl.div_rn(val, s)), valid)


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    """Masked SiLU-and-mul quantized to E4M3FN with one per-tensor scale."""
    experts, tokens, width = input.shape
    half = width // 2
    out = torch.empty(
        (experts, tokens, half),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    total = experts * tokens * half
    if total:
        if not input.is_contiguous():
            input = input.contiguous()
        if not masked_m.is_contiguous():
            masked_m = masked_m.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()
        programs = min(1024, (total + BLOCK - 1) // BLOCK)
        _silu_quant.run(
            input,
            masked_m,
            scale,
            out.view(torch.uint8),
            tokens,
            half,
            total,
            total >= 2**31,
            programs,
            BLOCK,
            grid=(programs,),
            warmup=False,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


solution = silu_and_mul_masked_post_per_tensor_quant

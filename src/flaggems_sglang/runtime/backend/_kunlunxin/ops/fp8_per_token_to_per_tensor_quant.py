# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128
# The XPU scratchpad rejects wider tiles, so the grid-stride loop carries
# the work instead of the tile.
BLOCK = 128


@triton.jit
def _decode_e4m3(byte):
    """Widen E4M3FN, with the layout the vendor compilers accept."""
    bits = byte.to(tl.int32)
    magnitude = bits & 127
    exponent, mantissa = magnitude >> 3, magnitude & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(
        tl.float32, bitcast=True
    )
    value = tl.where(
        exponent == 0, mantissa.to(tl.float32) * (1.0 / 512), normal
    )
    value = tl.where(magnitude == 127, float("nan"), value)
    return tl.where((bits & 128) != 0, -value, value)


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN, with signed words throughout."""
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
def _requantize(
    X,
    S,
    M,
    OS,
    Y,
    T: tl.constexpr,
    K: tl.constexpr,
    TOTAL: tl.constexpr,
    SPAN: tl.constexpr,
    XE: tl.constexpr,
    XT: tl.constexpr,
    SE: tl.constexpr,
    ST: tl.constexpr,
    YE: tl.constexpr,
    YT: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Laid out like the Ascend kernels that already compile on this chip: a
    # flat index, a one-dimensional grid and a grid-stride loop.
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    lane = tl.arange(0, BLOCK)
    for step in range(triton.cdiv(TOTAL, PROGRAMS * BLOCK)):
        index = (step * PROGRAMS + pid) * BLOCK + lane
        valid = index < TOTAL
        row, column = index // K, index % K
        expert, token = row // T, row % T
        kept = valid & (token < tl.load(M + expert, valid, 0))
        byte = tl.load(X + expert * XE + token * XT + column, kept, 0)
        scale = tl.load(
            S + expert * SE + token * ST + column // SPAN, kept, 0.0
        )
        inverse = 1.0 / tl.load(OS + index * 0, valid, 1.0)
        # Masked rows read code zero, which widens to zero and encodes back
        # to zero, so they store their zero bytes with no separate branch.
        value = _decode_e4m3(byte) * scale * inverse
        tl.store(
            Y + expert * YE + token * YT + column, _encode_fp8(value), valid
        )


def fp8_per_token_to_per_tensor_quant(x, x_scale, masked_m, output_scale):
    """Rescale per-token-group E4M3FN values onto one per-tensor scale."""
    experts, tokens, width = x.shape
    if width % GROUP:
        raise ValueError("the last dimension must be a multiple of 128")
    if x_scale.shape != (experts, tokens, width // GROUP):
        raise ValueError("x_scale must hold one scale per 128-wide group")
    if masked_m.shape != (experts,):
        raise ValueError("masked_m must hold one row count per expert")
    out = torch.empty_like(x)
    total = experts * tokens * width
    programs = min(1024, triton.cdiv(total, BLOCK))
    _requantize[(programs,)](
        x.view(torch.uint8),
        x_scale,
        masked_m,
        output_scale,
        out.view(torch.uint8),
        tokens,
        width,
        total,
        GROUP,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        out.stride(0),
        out.stride(1),
        total >= 2**31,
        programs,
        BLOCK,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

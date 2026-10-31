# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128


@triton.jit
def _decode_fp8(byte):
    """Widen E4M3FN bytes to FP32 with arithmetic only.

    The Ascend compiler rejects integer-to-float bitcasts, so the exponent
    and mantissa are recombined by scaling instead of by reassembling bits.
    """
    u = byte.to(tl.uint32)
    exponent = (u >> 3) & 15
    mantissa = (u & 7).to(tl.float32)
    # Subnormals drop the implicit leading one and share the exponent of
    # the smallest normal, so clamping the exponent covers both cases.
    leading = tl.where(exponent > 0, 8.0, 0.0)
    scale = tl.exp2(tl.maximum(exponent, 1).to(tl.float32) - 10.0)
    value = (mantissa + leading) * scale
    return tl.where((u & 0x80) != 0, -value, value)


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN, matching torch's round-to-nearest-even cast."""
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.minimum(tl.abs(value), 448.0)
    u = magnitude.to(tl.uint32, bitcast=True)
    rounded = u + 0x7FFFF + ((u >> 20) & 1)
    normal = (rounded >> 20) - (120 << 3)
    subnormal = (magnitude + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
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
    XE: tl.constexpr,
    XT: tl.constexpr,
    SE: tl.constexpr,
    ST: tl.constexpr,
    YE: tl.constexpr,
    YT: tl.constexpr,
    SPAN: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    expert = row // T
    token = row % T
    # Walking one scale group per step keeps each tile inside the vendor
    # scratchpad and the group scale a scalar, so no index arithmetic has
    # to divide a vector.
    kept = token < tl.load(M + expert)
    inverse = 1.0 / tl.load(OS)
    column = tl.arange(0, SPAN)
    source = X + expert * XE + token * XT
    scales = S + expert * SE + token * ST
    target = Y + expert * YE + token * YT
    for group in range(BLOCKS):
        byte = tl.load(source + group * SPAN + column, kept, other=0)
        scale = tl.load(scales + group)
        # Masked rows read zero, and zero survives the round trip, so they
        # store their zero bytes without a separate branch.
        value = _decode_fp8(byte) * scale * inverse
        tl.store(target + group * SPAN + column, _encode_fp8(value))


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
    _requantize[(experts * tokens,)](
        x.view(torch.uint8),
        x_scale,
        masked_m,
        output_scale,
        out.view(torch.uint8),
        tokens,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP,
        width // GROUP,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

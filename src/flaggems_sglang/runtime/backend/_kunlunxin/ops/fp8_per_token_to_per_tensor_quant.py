# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128
# The XPU scratchpad rejects the generic kernel's 1024-element tile, so a
# tile holds one scale group and the scale repeats across it by gather.
BLOCK = 128


@triton.jit
def _decode_fp8(byte):
    """Widen E4M3FN bytes to FP32 without a native FP8 cast."""
    u = byte.to(tl.uint32)
    sign = (u & 0x80) << 24
    magnitude = u & 0x7F
    normal = ((magnitude << 20) + 0x3C000000).to(tl.float32, bitcast=True)
    subnormal = normal * 2.0 - 2.0**-6
    value = tl.where(magnitude < 8, subnormal, normal)
    value = tl.where(magnitude == 0x7F, float("nan"), value)
    bits = value.to(tl.uint32, bitcast=True) | sign
    return bits.to(tl.float32, bitcast=True)


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
    K: tl.constexpr,
    XE: tl.constexpr,
    XT: tl.constexpr,
    SE: tl.constexpr,
    ST: tl.constexpr,
    YE: tl.constexpr,
    YT: tl.constexpr,
    SPAN: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    expert = row // T
    token = row % T
    column = tl.program_id(1) * TILE + tl.arange(0, TILE)
    live = column < K
    kept = token < tl.load(M + expert)
    byte = tl.load(X + expert * XE + token * XT + column, live & kept, other=0)
    scale = tl.load(
        S + expert * SE + token * ST + column // SPAN, live & kept, other=0.0
    )
    # Masked lanes read zero, and zero survives the round trip, so those
    # rows store their zero bytes without a separate branch.
    value = _decode_fp8(byte) * (scale * (1.0 / tl.load(OS)))
    tl.store(Y + expert * YE + token * YT + column, _encode_fp8(value), live)


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
    _requantize[(experts * tokens, triton.cdiv(width, BLOCK))](
        x.view(torch.uint8),
        x_scale,
        masked_m,
        output_scale,
        out.view(torch.uint8),
        tokens,
        width,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP,
        BLOCK,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

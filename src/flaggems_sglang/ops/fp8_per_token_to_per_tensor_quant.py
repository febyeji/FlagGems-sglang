# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128


@triton.jit
def _decode_fp8(byte):
    """Widen E4M3FN bytes to FP32 without a native FP8 cast."""
    u = byte.to(tl.uint32)
    sign = (u & 0x80) << 24
    magnitude = u & 0x7F
    # Placing the 4-bit exponent and 3-bit mantissa at the FP32 exponent and
    # mantissa boundary leaves an exponent short by the bias difference.
    normal = ((magnitude << 20) + 0x3C000000).to(tl.float32, bitcast=True)
    # Subnormals carry no implicit leading one, so undo it and rescale.
    subnormal = normal * 2.0 - 2.0**-6
    value = tl.where(magnitude < 8, subnormal, normal)
    value = tl.where(magnitude == 0x7F, float("nan"), value)
    # Carrying the sign as a bit keeps negative zero, which negation drops.
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
    # The bias has one FP8 subnormal step per FP32 ULP, so addition rounds
    # ties to even directly into the low mantissa bits.
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
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
):
    row = tl.program_id(0)
    expert = row // T
    token = row % T
    base = tl.program_id(1) * (TILE * SPAN)
    group = base // SPAN + tl.arange(0, TILE)
    column = base + tl.arange(0, TILE * SPAN)
    live = column < K
    out = Y + expert * YE + token * YT + column
    if token >= tl.load(M + expert):
        tl.store(out, tl.zeros((TILE * SPAN,), tl.uint8), live)
        return
    byte = tl.load(X + expert * XE + token * XT + column, live, other=0)
    scale = tl.load(
        S + expert * SE + token * ST + group, group < BLOCK, other=0.0
    )
    inverse = 1.0 / tl.load(OS)
    value = (
        tl.reshape(_decode_fp8(byte), (TILE, SPAN)) * scale[:, None] * inverse
    )
    tl.store(out, _encode_fp8(tl.reshape(value, (TILE * SPAN,))), live)


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
    blocks = width // GROUP
    tile = triton.next_power_of_2(min(blocks, 8) + 1) // 2
    _requantize[(experts * tokens, triton.cdiv(blocks, tile))](
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
        tile,
        blocks,
        GROUP,
        # One warp per 1024-element tile measured fastest across MoE shapes;
        # wider tiles win on K=4096 alone and lose on narrow and wide ones.
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

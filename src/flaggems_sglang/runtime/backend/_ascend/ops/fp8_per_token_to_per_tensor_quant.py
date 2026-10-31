# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128
BLOCK = 512
_WIDEN = {}


def _widen_table(device):
    """One FP32 value per E4M3FN code, laid out by code.

    The Ascend compiler rejects the bit arithmetic that widens FP8 inline,
    so the kernel reads the value from this constant instead. The entries
    come from the format definition: a 4-bit exponent biased by 7, a 3-bit
    mantissa, no implicit leading one below the smallest normal, and the
    all-ones pattern reserved for NaN.
    """
    cached = _WIDEN.get(device)
    if cached is None:
        values = []
        for code in range(256):
            exponent = (code >> 3) & 15
            mantissa = code & 7
            if exponent == 15 and mantissa == 7:
                value = float("nan")
            elif exponent == 0:
                value = mantissa * 2.0**-9
            else:
                value = (8 + mantissa) * 2.0 ** (exponent - 10)
            values.append(-value if code >> 7 else value)
        cached = torch.tensor(values, dtype=torch.float32, device=device)
        _WIDEN[device] = cached
    return cached


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
    W,
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
    kept = live & (token < tl.load(M + expert))
    byte = tl.load(X + expert * XE + token * XT + column, kept, other=0)
    scale = tl.load(
        S + expert * SE + token * ST + column // SPAN, kept, other=0.0
    )
    # Masked lanes read code zero, which widens to zero and encodes back to
    # zero, so those rows store their zero bytes without a separate branch.
    value = tl.load(W + byte.to(tl.int32)) * scale * (1.0 / tl.load(OS))
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
        _widen_table(x.device),
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
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

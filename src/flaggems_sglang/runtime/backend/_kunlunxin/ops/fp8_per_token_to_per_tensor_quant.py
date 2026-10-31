# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128
PACK = 4
# Words per program. The XPU scratchpad rejects wide tiles, and the chip
# runs a grid-stride loop slower than it runs more programs, so the tile
# stays narrow and the grid stays large.
BLOCK = 128


@triton.jit
def _decode_e4m3(code):
    """Widen E4M3FN with signed words and no native FP8 cast."""
    magnitude = code & 127
    exponent, mantissa = magnitude >> 3, magnitude & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(
        tl.float32, bitcast=True
    )
    value = tl.where(
        exponent == 0, mantissa.to(tl.float32) * (1.0 / 512), normal
    )
    value = tl.where(magnitude == 127, float("nan"), value)
    return tl.where((code & 128) != 0, -value, value)


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN, matching torch's round-to-nearest-even cast."""
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
    return tl.where(nan, 127, sign | encoded) & 0xFF


@triton.jit
def _lane(word, shift: tl.constexpr, scale):
    code = (word >> shift) & 0xFF
    return _encode_fp8(_decode_e4m3(code) * scale) << shift


@triton.jit
def _requantize(
    X,
    S,
    M,
    OS,
    Y,
    T: tl.constexpr,
    WORDS: tl.constexpr,
    TILES: tl.constexpr,
    SHIFT: tl.constexpr,
    XE: tl.constexpr,
    XT: tl.constexpr,
    SE: tl.constexpr,
    ST: tl.constexpr,
    YE: tl.constexpr,
    YT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Four E4M3FN codes ride in one 32-bit word, the way the batch-11
    # Kunlunxin kernel moves FP8, so the chip never touches a single byte.
    pid = tl.program_id(0)
    row = pid // TILES
    expert, token = row // T, row % T
    index = (pid % TILES) * BLOCK + tl.arange(0, BLOCK)
    live = index < WORDS
    kept = live & (token < tl.load(M + expert))
    word = tl.load(X + expert * XE + token * XT + index, kept, 0)
    scale = tl.load(
        S + expert * SE + token * ST + (index >> SHIFT), kept, 0.0
    ) * (1.0 / tl.load(OS))
    # Masked rows read code zero, which widens to zero and encodes back to
    # zero, so they store their zero bytes with no separate branch.
    packed = (
        _lane(word, 0, scale)
        | _lane(word, 8, scale)
        | _lane(word, 16, scale)
        | _lane(word, 24, scale)
    )
    tl.store(Y + expert * YE + token * YT + index, packed, live)


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
    bytes_in = x.view(torch.uint8)
    bytes_out = out.view(torch.uint8)
    if bytes_in.stride(0) % PACK or bytes_in.stride(1) % PACK:
        raise ValueError("rows must start on a 4-byte boundary")
    words = width // PACK
    tiles = triton.cdiv(words, BLOCK)
    _requantize[(experts * tokens * tiles,)](
        bytes_in.view(torch.int32),
        x_scale,
        masked_m,
        output_scale,
        bytes_out.view(torch.int32),
        tokens,
        words,
        tiles,
        (GROUP // PACK).bit_length() - 1,
        bytes_in.stride(0) // PACK,
        bytes_in.stride(1) // PACK,
        x_scale.stride(0),
        x_scale.stride(1),
        bytes_out.stride(0) // PACK,
        bytes_out.stride(1) // PACK,
        BLOCK,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

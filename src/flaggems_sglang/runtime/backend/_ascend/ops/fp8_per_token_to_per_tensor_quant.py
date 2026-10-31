# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

GROUP = 128
PACK = 4
BLOCK = 128
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
    """Round to E4M3FN, matching torch's round-to-nearest-even cast.

    Every integer here is signed. The batch-11 Ascend kernel that scores on
    this chip is the generic one with each unsigned word replaced by a
    signed one, and the magnitude stays non-negative, so the arithmetic
    shifts behave like logical ones.
    """
    bits = value.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.minimum(tl.abs(value), 448.0)
    u = magnitude.to(tl.int32, bitcast=True)
    rounded = u + 0x7FFFF + ((u >> 20) & 1)
    normal = (rounded >> 20) - (120 << 3)
    subnormal = (magnitude + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    encoded = tl.where(magnitude < 2.0**-6, subnormal, normal)
    return tl.where(nan, 127, sign | encoded)


@triton.jit
def _lane(word, shift: tl.constexpr, W, scale):
    code = ((word >> shift) & 0xFF).to(tl.int32)
    return _encode_fp8(tl.load(W + code) * scale) << shift


@triton.jit
def _requantize(
    X,
    S,
    M,
    OS,
    W,
    Y,
    TILES: tl.constexpr,
    T: tl.constexpr,
    WORDS: tl.constexpr,
    XE: tl.constexpr,
    XT: tl.constexpr,
    SE: tl.constexpr,
    ST: tl.constexpr,
    YE: tl.constexpr,
    YT: tl.constexpr,
    PER: tl.constexpr,
    TILE: tl.constexpr,
):
    # Four E4M3FN codes travel as one 32-bit word, so the kernel issues no
    # 8-bit load; a 1-D grid keeps every row index scalar.
    pid = tl.program_id(0)
    row = pid // TILES
    expert = row // T
    token = row % T
    index = (pid % TILES) * TILE + tl.arange(0, TILE)
    live = index < WORDS
    lane = tl.zeros((TILE,), tl.int32)
    rows = tl.load(M + (lane + expert))
    inverse = 1.0 / tl.load(OS + lane)
    kept = live & (token < rows)
    word = tl.load(X + expert * XE + token * XT + index, kept, other=0)
    scale = tl.load(
        S + expert * SE + token * ST + index // PER, kept, other=0.0
    )
    # Masked lanes read code zero, which widens to zero and encodes back to
    # zero, so those rows store their zero bytes without a separate branch.
    scale = scale * inverse
    packed = (
        _lane(word, 0, W, scale)
        | _lane(word, 8, W, scale)
        | _lane(word, 16, W, scale)
        | _lane(word, 24, W, scale)
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
        _widen_table(x.device),
        bytes_out.view(torch.int32),
        tiles,
        tokens,
        words,
        bytes_in.stride(0) // PACK,
        bytes_in.stride(1) // PACK,
        x_scale.stride(0),
        x_scale.stride(1),
        bytes_out.stride(0) // PACK,
        bytes_out.stride(1) // PACK,
        GROUP // PACK,
        BLOCK,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = fp8_per_token_to_per_tensor_quant

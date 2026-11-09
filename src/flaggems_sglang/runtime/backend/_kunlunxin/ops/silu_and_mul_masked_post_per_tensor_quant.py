# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]

# Words per program. The XPU scratchpad rejects wide tiles, and the chip
# runs a grid-stride loop slower than it runs more programs, so the tile
# stays narrow and the grid stays large.
BLOCK = 128


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN, matching torch's saturating round-to-nearest cast."""
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
def _lane(X, base, column, kept, H: tl.constexpr, s):
    gate = tl.load(X + base + column, kept, other=0.0).to(tl.float32)
    up = tl.load(X + base + H + column, kept, other=0.0).to(tl.float32)
    val = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
    return _encode_fp8(tl.div_rn(val, s))


@triton.jit
def _silu_quant(
    X,
    M,
    S,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    WORDS: tl.constexpr,
    TILES: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program takes one row's tile of output words. Four E4M3FN codes
    # ride in one 32-bit word, the way the batch-12 Kunlunxin kernel moves
    # FP8, so the chip never touches a single byte; the four lanes of a word
    # come from four strided one-dimensional reads, since a two-dimensional
    # gather does not compile here. A row length that is no multiple of four
    # takes the byte path instead, which is slow on this chip but exact.
    pid = tl.program_id(0)
    row = pid // TILES
    tile = pid - row * TILES
    expert = row // T
    token = row - expert * T
    n = tl.load(M + expert)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    live = index < WORDS
    # Masked rows read zeros and so encode to a zero byte, which is what the
    # reference leaves there.
    kept = live & (token < n)
    s = tl.load(S)
    base = row.to(tl.int64) * (2 * H)
    if PACK == 4:
        word = (
            _lane(X, base, index * 4, kept, H, s)
            | (_lane(X, base, index * 4 + 1, kept, H, s) << 8)
            | (_lane(X, base, index * 4 + 2, kept, H, s) << 16)
            | (_lane(X, base, index * 4 + 3, kept, H, s) << 24)
        )
        tl.store(Y + row.to(tl.int64) * WORDS + index, word, live)
    else:
        byte = _lane(X, base, index, kept, H, s)
        tl.store(Y + row.to(tl.int64) * H + index, byte.to(tl.uint8), live)


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    """Masked SiLU-and-mul quantized to E4M3FN with one per-tensor scale."""
    experts, tokens, width = input.shape
    half = width // 2
    out = torch.empty(
        (experts, tokens, half),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    if experts * tokens * half:
        if not input.is_contiguous():
            input = input.contiguous()
        if not masked_m.is_contiguous():
            masked_m = masked_m.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()
        pack = 4 if half % 4 == 0 else 1
        words = half // pack
        tiles = (words + BLOCK - 1) // BLOCK
        _silu_quant.run(
            input,
            masked_m,
            scale,
            out.view(torch.int32) if pack == 4 else out.view(torch.uint8),
            tokens,
            half,
            words,
            tiles,
            pack,
            BLOCK,
            grid=(experts * tokens * tiles,),
            warmup=False,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


solution = silu_and_mul_masked_post_per_tensor_quant

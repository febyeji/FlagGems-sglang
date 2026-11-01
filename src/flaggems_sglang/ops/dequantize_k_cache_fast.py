# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["dequantize_k_cache_fast"]

# The packed row is fixed by the model: 512 quantized nope channels, one
# FP32 scale per group of them, then 64 rope channels kept in BF16.
DIM_NOPE = 512
DIM_ROPE = 64
WARPS = 4


@triton.jit
def _decode(byte):
    """Widen one E4M3FN byte to FP32 with integer arithmetic.

    The chips this runs on have no native FP8 cast, so the byte is rebuilt
    as an FP32 word instead of converted.
    """
    # The byte arrives in a signed word, so the mask is what makes it a
    # number again: these chips are the ones that dislike unsigned types.
    code = byte.to(tl.int32) & 0xFF
    payload = code & 0x7F
    # Dropping exponent and mantissa into an FP32 word leaves the value
    # short by the difference of the two biases, 2^120, and adding that to
    # the exponent field is the same scaling without touching the mantissa.
    normal = ((payload << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    # Under the smallest normal the exponent is fixed, so the payload is
    # the whole mantissa and the value is a count of 2^-9.
    magnitude = tl.where(
        payload < 8, payload.to(tl.float32) * 0.001953125, normal
    )
    # E4M3FN spends its one reserved pattern on NaN and has no infinity.
    magnitude = tl.where(payload == 0x7F, float("nan"), magnitude)
    return tl.where(code >= 128, -magnitude, magnitude)


@triton.jit
def _dequantize(
    OUT,
    NOPE,
    SCALE,
    ROPE,
    OUT_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    SCALE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    SCALE_OFF: tl.constexpr,
    ROPE_OFF: tl.constexpr,
    GROUP: tl.constexpr,
    TILES: tl.constexpr,
    ROPE_W: tl.constexpr,
):
    # One program per row. The row divides into whole tiles and the grid
    # into whole rows, so every index here is affine and nothing is masked.
    token = tl.program_id(0)
    tile = tl.arange(0, TILES)
    column = tl.arange(0, GROUP)
    # A tile per row of the block, which gives each one its own scale
    # without gathering it per channel.
    offset = tile[:, None] * GROUP + column[None, :]
    byte = tl.load(NOPE + token * NOPE_S0 + offset)
    scale = tl.load(SCALE + token * SCALE_S0 + SCALE_OFF + tile)
    tl.store(
        OUT + token * OUT_S0 + offset,
        (_decode(byte) * scale[:, None]).to(tl.bfloat16),
    )
    # The rope tail is already BF16 inside the FP8 row, so it is a copy.
    tail = tl.arange(0, ROPE_W)
    rope = tl.load(ROPE + token * ROPE_S0 + ROPE_OFF + tail)
    tl.store(OUT + token * OUT_S0 + TILES * GROUP + tail, rope)


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    """Unpack an MLA K cache row: scale the nope tiles, copy the rope tail.

    The scale and rope regions are reinterpreted rather than converted, so
    the kernel reads the same rows through an FP32 and a BF16 view beside
    the byte view it decodes.
    """
    rows = quant_k_cache.shape[0]
    tiles = DIM_NOPE // group_size
    out = torch.empty(
        (rows, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if rows:
        stride = quant_k_cache.stride(0) * quant_k_cache.element_size()
        _dequantize[(rows,)](
            out,
            triton.reinterpret(quant_k_cache, tl.int8),
            triton.reinterpret(quant_k_cache, tl.float32),
            triton.reinterpret(quant_k_cache, tl.bfloat16),
            out.stride(0),
            stride,
            stride // 4,
            stride // 2,
            DIM_NOPE // 4,
            (DIM_NOPE + tiles * 4) // 2,
            group_size,
            tiles,
            DIM_ROPE,
            num_warps=WARPS,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


solution = dequantize_k_cache_fast

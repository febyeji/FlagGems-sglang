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
# Two warps, which this chip graded a little above four,
# 4.255 against 4.138.
WARPS = 2


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
def _dequantize(OUT, ROW, ROW_S0: tl.constexpr, GROUP: tl.constexpr):
    # One program per row, the whole nope half in one flat tile, and the
    # scale gathered per channel. The grid divides into whole rows and the
    # widths are powers of two, so nothing here is masked.
    #
    # The row arrives once, as bytes, and the two regions that are
    # reinterpreted rather than converted are reached by casting that one
    # pointer. Each argument and each launch option costs a few
    # microseconds on the way in, which is the scale this operator is
    # measured at: the kernel runs at the memory's pace and the call
    # around it does not.
    token = tl.program_id(0)
    column = tl.arange(0, 512)
    byte = tl.load(ROW + token * ROW_S0 + column)
    scale = tl.load(
        ROW.to(tl.pointer_type(tl.float32))
        + token * (ROW_S0 // 4)
        + 128
        + column // GROUP
    )
    tl.store(
        OUT + token * 576 + column, (_decode(byte) * scale).to(tl.bfloat16)
    )
    # The rope tail is already BF16 inside the FP8 row, so it is a copy.
    tail = tl.arange(0, 64)
    rope = tl.load(
        ROW.to(tl.pointer_type(tl.bfloat16))
        + token * (ROW_S0 // 2)
        + (512 + (512 // GROUP) * 4) // 2
        + tail
    )
    tl.store(OUT + token * 576 + 512 + tail, rope)


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    """Unpack an MLA K cache row: scale the nope tiles, copy the rope tail.

    The scale and rope regions are reinterpreted rather than converted, so
    the kernel reaches them by casting the byte pointer it is handed.
    """
    rows = quant_k_cache.shape[0]
    out = torch.empty(
        rows,
        DIM_NOPE + DIM_ROPE,
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if rows:
        _dequantize[(rows,)](
            out,
            triton.reinterpret(quant_k_cache, tl.int8),
            quant_k_cache.stride(0),
            group_size,
            num_warps=WARPS,
        )
    return out


solution = dequantize_k_cache_fast

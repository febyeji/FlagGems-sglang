# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fp8_quantize"]


@triton.jit
def _offset(
    index, SHAPE: tl.constexpr, STRIDES: tl.constexpr, CONTIG: tl.constexpr
):
    if CONTIG:
        return index
    rest = index
    offset = tl.full(index.shape, 0, index.dtype)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (rest % SHAPE[axis]) * STRIDES[axis]
        rest = rest // SHAPE[axis]
    return offset


@triton.jit
def _load(pointer, mask):
    if pointer.dtype.element_ty == tl.bfloat16:
        word = tl.load(pointer.to(tl.pointer_type(tl.uint16)), mask, other=0)
        return (word.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(pointer, mask, other=0).to(tl.float32)


@triton.jit
def _encode_fp8(value, E5: tl.constexpr):
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    if E5:
        M: tl.constexpr = 2
        BIAS: tl.constexpr = 15
        MAX: tl.constexpr = 57344.0
        MIN_NORMAL: tl.constexpr = 2.0**-14
        ROUND_BIAS: tl.constexpr = 128.0
        ROUND_BITS: tl.constexpr = 0x43000000
    else:
        M: tl.constexpr = 3
        BIAS: tl.constexpr = 7
        MAX: tl.constexpr = 448.0
        MIN_NORMAL: tl.constexpr = 2.0**-6
        ROUND_BIAS: tl.constexpr = 16384.0
        ROUND_BITS: tl.constexpr = 0x46800000
    magnitude = tl.minimum(tl.abs(value), MAX)
    u = magnitude.to(tl.uint32, bitcast=True)
    shift: tl.constexpr = 23 - M
    rounded = u + ((1 << (shift - 1)) - 1) + ((u >> shift) & 1)
    normal = (rounded >> shift) - ((127 - BIAS) << M)
    # The bias has one FP8 subnormal step per FP32 ULP, so addition rounds
    # ties to even directly into the low mantissa bits.
    subnormal = (magnitude + ROUND_BIAS).to(
        tl.uint32, bitcast=True
    ) - ROUND_BITS
    encoded = tl.where(magnitude < MIN_NORMAL, subnormal, normal)
    return tl.where(nan, 127, sign | encoded).to(tl.uint8)


@triton.jit
def _quantize(
    X,
    Y,
    COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    E5: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    index = (pid * BLOCK + tl.arange(0, BLOCK)) * 4
    packed = tl.full((BLOCK,), 0, tl.uint32)
    for lane in tl.static_range(4):
        position = index + lane
        value = _load(
            X + _offset(position, SHAPE, STRIDES, CONTIG), position < COUNT
        )
        if SCALE != 1.0:
            value *= SCALE
        code = _encode_fp8(value, E5)
        packed |= code.to(tl.uint32) << (lane * 8)
        if COUNT % 4 != 0:
            tl.store(
                Y + position, code, (position < COUNT) & (index + 3 >= COUNT)
            )
    tl.store(
        Y.to(tl.pointer_type(tl.uint32)) + index // 4,
        packed,
        index + 3 < COUNT,
    )


def fp8_quantize(x, scale_inv, fp8_dtype):
    """Saturate and round to FP8, including on chips without native FP8 casts."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("fp8_dtype must be E4M3FN or E5M2")
    out = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
    count = x.numel()
    if count:
        bound = max(
            count + 4096, sum((s - 1) * d for s, d in zip(x.shape, x.stride()))
        )
        _quantize[(triton.cdiv(count, 1024),)](
            x,
            out.view(torch.uint8),
            count,
            scale_inv,
            fp8_dtype == torch.float8_e5m2,
            tuple(x.shape),
            tuple(x.stride()),
            x.is_contiguous(),
            bound >= 2**31,
            256,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


solution = fp8_quantize

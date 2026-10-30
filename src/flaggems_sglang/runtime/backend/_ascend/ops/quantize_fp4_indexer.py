# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["quantize_fp4_indexer"]


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
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _load(pointer, mask):
    return tl.load(pointer, mask, other=0).to(tl.float32)


@triton.jit
def _fp4_code(value):
    magnitude = tl.minimum(
        tl.abs(value), 6.0, propagate_nan=tl.PropagateNan.ALL
    )
    code = (magnitude > 0.25).to(tl.int32)
    code += (magnitude > 0.75).to(tl.int32)
    code += (magnitude > 1.25).to(tl.int32)
    code += (magnitude > 1.75).to(tl.int32)
    code += (magnitude > 2.5).to(tl.int32)
    code += (magnitude > 3.5).to(tl.int32)
    code += (magnitude > 5.0).to(tl.int32)
    return code | (((value < 0.0) & (code != 0)).to(tl.int32) << 3)


@triton.jit
def _quant_fp4(
    X,
    Q,
    S,
    COUNT: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    col = tl.arange(0, 16)
    for step in range(triton.cdiv(COUNT, PROGRAMS * ROWS)):
        first = (step * PROGRAMS + pid) * ROWS
        group = first * 4 + tl.arange(0, ROWS * 4)
        index = group[:, None] * 32 + col[None, :] * 2
        valid = group[:, None] < COUNT * 4
        low = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), valid)
        high = _load(X + _offset(index + 1, SHAPE, STRIDES, CONTIG), valid)
        maximum = tl.reduce(_max_nan(tl.abs(low), tl.abs(high)), 1, _max_nan)
        sf = tl.maximum(maximum / 6.0, 1e-4, propagate_nan=tl.PropagateNan.ALL)
        bits = sf.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exponent = tl.minimum(tl.maximum(exponent, 1), 254)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        low_code = _fp4_code(tl.div_rn(low, scale[:, None]))
        high_code = _fp4_code(tl.div_rn(high, scale[:, None]))
        packed = (low_code | (high_code << 4)).to(tl.uint8)
        tl.store(Q + group[:, None] * 16 + col[None, :], packed, valid)
        exponents = tl.reshape(exponent, (ROWS, 4))
        packed_scale = tl.sum(exponents << (8 * tl.arange(0, 4)[None, :]), 1)
        row = first + tl.arange(0, ROWS)
        tl.store(S + row, packed_scale, row < COUNT)


def quantize_fp4_indexer(x):
    """Pack E2M1 codes and four UE8M0 scales for each 128-element token."""
    if x.shape[-1] != 128:
        raise ValueError("The last dimension must contain 128 elements")
    count = x.numel() // 128
    q = torch.empty((count, 64), device=x.device, dtype=torch.int8)
    scales = torch.empty((count,), device=x.device, dtype=torch.int32)
    if count:
        rows = 4
        programs = triton.cdiv(count, rows)
        programs = min(40, programs)
        bound = max(
            x.numel() + 128 * rows,
            sum((s - 1) * d for s, d in zip(x.shape, x.stride())),
        )
        _quant_fp4[(programs,)](
            x,
            q.view(torch.uint8),
            scales,
            count,
            tuple(x.shape),
            tuple(x.stride()),
            x.is_contiguous(),
            bound >= 2**31,
            programs,
            rows,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return q, scales


solution = quantize_fp4_indexer

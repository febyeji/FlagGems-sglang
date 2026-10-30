# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["per_token_quant_int8_sum"]


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
def _quant_int8(
    X,
    Q,
    S,
    SUM,
    COUNT: tl.constexpr,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    CAL_SUM: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    col = tl.arange(0, BLOCK)
    for step in range(triton.cdiv(COUNT, PROGRAMS * ROWS)):
        row = (step * PROGRAMS + pid) * ROWS + tl.arange(0, ROWS)
        index = row[:, None] * N + col[None, :]
        valid = (row[:, None] < COUNT) & (col[None, :] < N)
        value = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), valid)
        maximum = tl.maximum(tl.max(tl.abs(value), 1), 1e-10)
        scale = maximum / 127.0
        inverse = tl.div_rn(1.0, maximum) * 127.0
        scaled = value * inverse[:, None]
        # Truncation of the nonnegative magnitude implements half-away rounding.
        rounded = (tl.abs(scaled) + 0.5).to(tl.int32)
        code = tl.where(scaled < 0.0, -rounded, rounded).to(tl.int8)
        tl.store(Q + index, code, valid)
        tl.store(S + row, scale, row < COUNT)
        if CAL_SUM:
            tl.store(SUM + row, tl.sum(value, 1), row < COUNT)


def per_token_quant_int8_sum(x, scale_dtype, cal_sum):
    """Quantize rows to INT8, optionally accumulating their sums in the same pass."""
    n = x.shape[-1]
    if n <= 0:
        raise ValueError("The last dimension must be nonempty")
    count = x.numel() // n
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scales = torch.empty(
        x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype
    )
    sums = (
        torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)
        if cal_sum
        else scales
    )
    if count:
        block = triton.next_power_of_2(n)
        rows = min(4, max(1, 1024 // block))
        programs = triton.cdiv(count, rows)
        # One tile per program on this backend.
        bound = max(
            x.numel() + block * rows,
            sum((s - 1) * d for s, d in zip(x.shape, x.stride())),
        )
        _quant_int8[(programs,)](
            x,
            q,
            scales,
            sums,
            count,
            n,
            tuple(x.shape),
            tuple(x.stride()),
            x.is_contiguous(),
            cal_sum,
            bound >= 2**31,
            programs,
            rows,
            block,
            num_warps=8 if block >= 4096 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return (q, scales, sums) if cal_sum else (q, scales)


solution = per_token_quant_int8_sum

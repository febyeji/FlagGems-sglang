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
    return tl.load(pointer, mask, other=0).to(tl.float32)


@triton.jit
def _quant_int8(
    X,
    Q,
    S,
    SUM,
    COUNT: tl.constexpr,
    PROGRAMS: tl.constexpr,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    CAL_SUM: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    for row in range(pid, COUNT, PROGRAMS):
        col = tl.arange(0, BLOCK)
        index = row * N + col
        value = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), col < N)
        maximum = tl.maximum(tl.max(tl.abs(value), 0), 1e-10)
        scale = maximum / 127.0
        inverse = (1.0 / maximum) * 127.0
        scaled = value * inverse
        # Truncation after a signed half-unit bias rounds ties away from zero.
        code = (scaled + tl.where(scaled < 0.0, -0.5, 0.5)).to(tl.int8)
        tl.store(Q + index, code, col < N)
        tl.store(S + row, scale)
        if CAL_SUM:
            tl.store(SUM + row, tl.sum(value, 0))


def per_token_quant_int8_sum(x, scale_dtype, cal_sum):
    """Quantize rows in a bounded Ascend launch and optionally return sums."""
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
        programs = min(40, count)
        bound = max(
            x.numel() + block,
            sum((s - 1) * d for s, d in zip(x.shape, x.stride())),
        )
        _quant_int8[(programs,)](
            x,
            q,
            scales,
            sums,
            count,
            programs,
            n,
            tuple(x.shape),
            tuple(x.stride()),
            x.is_contiguous(),
            cal_sum,
            bound >= 2**31,
            block,
            num_warps=8,
            num_stages=2 if count > programs else 1,
            enable_fp_fusion=False,
        )
    return (q, scales, sums) if cal_sum else (q, scales)


solution = per_token_quant_int8_sum

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["per_tensor_absmax_fp8"]


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
def _partial_max(
    X,
    P,
    COUNT: tl.constexpr,
    PARTS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    maximum = tl.full((BLOCK,), 0.0, tl.float32)
    for step in range(triton.cdiv(COUNT, PARTS * BLOCK)):
        index = (step * PARTS + pid) * BLOCK + tl.arange(0, BLOCK)
        x = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), index < COUNT)
        value = tl.where(index < COUNT, tl.abs(x), 0.0)
        maximum = tl.maximum(maximum, value)
    value = tl.max(maximum, 0)
    if FINAL:
        value = value / 448.0
    tl.store(P + pid, value)


@triton.jit
def _finish(P, S, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.load(P + i, i < PARTS, 0)
    tl.store(S, tl.max(value, 0) / 448.0)


def per_tensor_absmax_fp8(input, output_s):
    """Reduce disjoint blocks, then write the maximum scale once."""
    count = input.numel()
    if count == 0:
        raise ValueError("amax requires a nonempty tensor")
    if output_s.numel() != 1 or output_s.dtype != torch.float32:
        raise ValueError("output_s must contain one fp32 element")
    parts = min(1024, triton.cdiv(count, 4096))
    partial = (
        torch.empty((parts,), dtype=torch.float32, device=input.device)
        if parts > 1
        else output_s
    )
    bound = max(
        count + 4096,
        sum((s - 1) * d for s, d in zip(input.shape, input.stride())),
    )
    _partial_max[(parts,)](
        input,
        partial,
        count,
        parts,
        tuple(input.shape),
        tuple(input.stride()),
        input.is_contiguous(),
        bound >= 2**31,
        parts == 1,
        4096,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if parts > 1:
        _finish[(1,)](
            partial,
            output_s,
            parts,
            triton.next_power_of_2(parts),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output_s


solution = per_tensor_absmax_fp8

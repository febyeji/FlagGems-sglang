# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["per_token_group_quant_8bit_v2"]


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
def _group_quant(
    X,
    Q,
    S,
    GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    EPS: tl.constexpr,
    MIN8: tl.constexpr,
    MAX8: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    QSTRIDES: tl.constexpr,
    SSHAPE: tl.constexpr,
    SSTRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    QCONTIG: tl.constexpr,
    SCONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    col = tl.arange(0, BLOCK)
    for step in range(triton.cdiv(GROUPS, PROGRAMS * ROWS)):
        group = (step * PROGRAMS + pid) * ROWS + tl.arange(0, ROWS)
        index = group[:, None] * GROUP + col[None, :]
        valid = (group[:, None] < GROUPS) & (col[None, :] < GROUP)
        value = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), valid)
        maximum = tl.maximum(
            tl.reduce(tl.abs(value), 1, _max_nan),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = maximum / MAX8
        value = tl.div_rn(value, scale[:, None])
        value = tl.minimum(
            tl.maximum(value, MIN8, propagate_nan=tl.PropagateNan.ALL),
            MAX8,
            propagate_nan=tl.PropagateNan.ALL,
        )
        code = _encode_fp8(value, False)
        tl.store(Q + _offset(index, SHAPE, QSTRIDES, QCONTIG), code, valid)
        tl.store(
            S + _offset(group, SSHAPE, SSTRIDES, SCONTIG),
            scale,
            group < GROUPS,
        )


def _launch(input, output_q, output_s, group_size, eps, min_8bit, max_8bit):
    if group_size <= 0 or input.shape[-1] % group_size:
        raise ValueError("The last dimension must be divisible by group_size")
    if output_q.shape != input.shape or output_q.dtype != torch.float8_e4m3fn:
        raise ValueError("output_q must be E4M3FN with the input shape")
    groups = input.numel() // group_size
    if output_s.numel() != groups or output_s.dtype != torch.float32:
        raise ValueError("output_s must contain one fp32 scale per group")
    if groups:
        block = triton.next_power_of_2(group_size)
        rows = min(4, max(1, 1024 // block))
        programs = triton.cdiv(groups, rows)
        # One tile per program on this backend.
        bound = max(
            [input.numel() + block * rows]
            + [
                sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
                for t in (input, output_q, output_s)
            ]
        )
        _group_quant[(programs,)](
            input,
            output_q.view(torch.uint8),
            output_s,
            groups,
            group_size,
            eps,
            min_8bit,
            max_8bit,
            tuple(input.shape),
            tuple(input.stride()),
            tuple(output_q.stride()),
            tuple(output_s.shape),
            tuple(output_s.stride()),
            input.is_contiguous(),
            output_q.is_contiguous(),
            output_s.is_contiguous(),
            bound >= 2**31,
            programs,
            rows,
            block,
            num_warps=8 if block >= 4096 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output_q, output_s


def per_token_group_quant_8bit_v2(
    input, output_q, output_s, group_size, eps, min_8bit, max_8bit
):
    """Write groupwise E4M3FN codes and fp32 scales into the supplied outputs."""
    return _launch(
        input, output_q, output_s, group_size, eps, min_8bit, max_8bit
    )


solution = per_token_group_quant_8bit_v2

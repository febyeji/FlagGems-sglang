# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["per_tensor_quant_fp8"]


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
    for start in range(pid * BLOCK, COUNT, PARTS * BLOCK):
        index = start + tl.arange(0, BLOCK)
        x = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), index < COUNT)
        value = tl.where(index < COUNT, tl.abs(x), 0.0)
        maximum = tl.maximum(maximum, value)
    value = tl.max(maximum, 0)
    if FINAL:
        value = value / 448.0
    tl.store(P + pid, value)


@triton.jit
def _quant_tensor(
    X,
    Q,
    S,
    P,
    COUNT: tl.constexpr,
    PARTS: tl.constexpr,
    MODE: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    QSTRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    QCONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    value = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), index < COUNT)
    if MODE == 1:
        # Small tensors fit in a local reduction. Replicating it across a few
        # programs avoids an extra launch and distributes the FP8 conversion.
        r = tl.arange(0, REDUCE)
        all_values = _load(X + _offset(r, SHAPE, STRIDES, CONTIG), r < COUNT)
        scale = tl.max(tl.where(r < COUNT, tl.abs(all_values), 0.0), 0) / 448.0
    elif MODE == 2:
        p = tl.arange(0, REDUCE)
        partial = tl.load(P + tl.minimum(p, PARTS - 1))
        scale = tl.max(tl.where(p < PARTS, partial, 0.0), 0) / 448.0
    else:
        scale = tl.load(S)
    # Unstored lanes need no zero division, which is costly on some targets.
    value = tl.where(index < COUNT, value, 1.0)
    quantized = _encode_fp8(tl.div_rn(value, scale), False)
    offset = _offset(index, SHAPE, QSTRIDES, QCONTIG)
    tl.store(Q + offset, quantized, index < COUNT)
    if MODE != 0:
        if tl.program_id(0) == 0:
            tl.store(S, scale)


def per_tensor_quant_fp8(input, output_q, output_s, is_static):
    """Write a static or dynamically scaled E4M3FN quantization in place."""
    count = input.numel()
    if count == 0:
        raise ValueError("quantization requires a nonempty tensor")
    if output_q.shape != input.shape or output_q.dtype != torch.float8_e4m3fn:
        raise ValueError("output_q must be E4M3FN with the input shape")
    if output_s.numel() != 1 or output_s.dtype != torch.float32:
        raise ValueError("output_s must contain one fp32 element")
    bound = max(
        [count + 4096]
        + [
            sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
            for t in (input, output_q)
        ]
    )
    mode = 0 if is_static else (1 if count <= 16384 else 2)
    # Smaller tiles expose more parallelism on medium tensors. Dense 16-bit
    # inputs retain larger tiles once partial-reduction overhead dominates.
    small_reduce = (
        mode == 2
        and count <= 262144
        and (
            count <= 131072
            or not input.is_contiguous()
            or input.element_size() == 4
        )
    )
    reduce_block = 1024 if small_reduce else 4096
    parts = min(1024, triton.cdiv(count, reduce_block))
    partial = output_s
    if mode == 2:
        partial = torch.empty(
            (parts,), dtype=torch.float32, device=input.device
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
            False,
            reduce_block,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    block = 256 if mode == 1 and count <= 8192 else 1024
    if small_reduce:
        block = 512
    warps = 8 if mode == 1 and count > 8192 else 4
    _quant_tensor[(triton.cdiv(count, block),)](
        input,
        output_q.view(torch.uint8),
        output_s,
        partial,
        count,
        parts,
        mode,
        tuple(input.shape),
        tuple(input.stride()),
        tuple(output_q.stride()),
        input.is_contiguous(),
        output_q.is_contiguous(),
        bound >= 2**31,
        triton.next_power_of_2(count if mode == 1 else parts),
        block,
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return output_q, output_s


solution = per_tensor_quant_fp8

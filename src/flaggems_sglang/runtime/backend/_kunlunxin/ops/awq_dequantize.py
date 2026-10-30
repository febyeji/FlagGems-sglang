# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["awq_dequantize"]


@triton.jit
def _dequantize(
    W,
    S,
    Z,
    Y,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    Z0: tl.constexpr,
    Z1: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    if Y.dtype.element_ty == tl.float32:
        index = pid * BLOCK + tl.arange(0, BLOCK)
        row, col = index // N, index % N
        shift = ((col % 8) // 2 + (col % 2) * 4) * 4
        word = tl.load(W + row * W0 + (col // 8) * W1, index < K * N, 0)
        zero = tl.load(
            Z + (row // GROUP) * Z0 + (col // 8) * Z1, index < K * N, 0
        )
        scale = tl.load(
            S + (row // GROUP) * S0 + col * S1, index < K * N, 0
        ).to(tl.float32)
        value = (((word >> shift) & 15) - ((zero >> shift) & 15)).to(
            tl.float32
        ) * scale
        tl.store(Y + index, value, index < K * N)
    else:
        packed = tl.full((BLOCK,), 0, tl.uint32)
        for lane in tl.static_range(2):
            index = (pid * BLOCK + tl.arange(0, BLOCK)) * 2 + lane
            row, col = index // N, index % N
            shift = ((col % 8) // 2 + (col % 2) * 4) * 4
            word = tl.load(W + row * W0 + (col // 8) * W1, index < K * N, 0)
            zero = tl.load(
                Z + (row // GROUP) * Z0 + (col // 8) * Z1, index < K * N, 0
            )
            scale = tl.load(
                S + (row // GROUP) * S0 + col * S1, index < K * N, 0
            ).to(tl.float32)
            value = (((word >> shift) & 15) - ((zero >> shift) & 15)).to(
                tl.float32
            ) * scale
            code = value.to(Y.dtype.element_ty).to(tl.uint16, bitcast=True)
            packed |= code.to(tl.uint32) << (lane * 16)
        output_index = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            Y.to(tl.pointer_type(tl.uint32)) + output_index,
            packed,
            output_index * 2 < K * N,
        )


def awq_dequantize(qweight, scales, zeros):
    """Unpack AWQ's interleaved nibbles and apply per-group affine scales."""
    k, packed_n = qweight.shape
    n = packed_n * 8
    groups = scales.shape[0]
    if groups == 0 or k % groups or scales.shape != (groups, n):
        raise ValueError("scales must divide the K dimension and match N")
    if zeros.shape != (groups, packed_n):
        raise ValueError("zeros must have shape [groups, N // 8]")
    out = torch.empty((k, n), device=qweight.device, dtype=scales.dtype)
    if out.numel():
        bound = max(
            [k * n + 4096]
            + [
                sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
                for t in (qweight, scales, zeros)
            ]
        )
        block = 256 if k * n <= 32768 else 1024
        width = 1 if scales.dtype == torch.float32 else 2
        _dequantize[(triton.cdiv(k * n, block * width),)](
            qweight,
            scales,
            zeros,
            out,
            k,
            n,
            k // groups,
            *qweight.stride(),
            *scales.stride(),
            *zeros.stride(),
            bound >= 2**31,
            block,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


solution = awq_dequantize

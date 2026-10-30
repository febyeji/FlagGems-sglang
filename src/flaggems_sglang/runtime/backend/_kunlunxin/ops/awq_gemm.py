# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["awq_gemm"]


@triton.jit
def _load(pointer, mask):
    if pointer.dtype.element_ty == tl.bfloat16:
        word = tl.load(pointer.to(tl.pointer_type(tl.uint16)), mask, other=0)
        return (word.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(pointer, mask, other=0).to(tl.float32)


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
    index = pid * BLOCK + tl.arange(0, BLOCK)
    row, col = index % K, index // K
    shift = ((col % 8) // 2 + (col % 2) * 4) * 4
    word = tl.load(W + row * W0 + (col // 8) * W1, index < K * N, 0)
    zero = tl.load(Z + (row // GROUP) * Z0 + (col // 8) * Z1, index < K * N, 0)
    scale = _load(S + (row // GROUP) * S0 + col * S1, index < K * N)
    value = (((word >> shift) & 15) - ((zero >> shift) & 15)).to(
        tl.float32
    ) * scale
    tl.store(Y + index, value.to(S.dtype.element_ty), index < K * N)


@triton.jit
def _dot_load(pointer, mask):
    value = _load(pointer, mask)
    if pointer.dtype.element_ty == tl.float32:
        bits = value.to(tl.int32, bitcast=True)
        rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & -8192
        value = rounded.to(tl.float32, bitcast=True)
    return value


@triton.jit
def _dense(
    A,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    WIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile, part = tl.program_id(0), tl.program_id(1)
    if WIDE:
        tile, part = tile.to(tl.int64), part.to(tl.int64)
    row = tile // tl.cdiv(N, BN) * BM
    cols = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    ks = part * BK + tl.arange(0, BK)
    if BM > 0:
        acc0 = tl.full((BN,), 0, tl.float32)
    if BM > 1:
        acc1 = tl.full((BN,), 0, tl.float32)
    if BM > 2:
        acc2 = tl.full((BN,), 0, tl.float32)
    if BM > 3:
        acc3 = tl.full((BN,), 0, tl.float32)
    if BM > 4:
        acc4 = tl.full((BN,), 0, tl.float32)
    if BM > 5:
        acc5 = tl.full((BN,), 0, tl.float32)
    if BM > 6:
        acc6 = tl.full((BN,), 0, tl.float32)
    if BM > 7:
        acc7 = tl.full((BN,), 0, tl.float32)
    for step in tl.static_range((K + BK * SPLIT - 1) // (BK * SPLIT)):
        valid = (cols[:, None] < N) & (ks[None, :] < K)
        w = _dot_load(W + cols[:, None] * K + ks[None, :], valid)
        if BM > 0:
            a0 = _dot_load(
                A + (row + 0) * A0 + ks * A1, (row + 0 < M) & (ks < K)
            )
            acc0 += tl.sum(tl.where(valid, a0[None, :] * w, 0.0), 1)
        if BM > 1:
            a1 = _dot_load(
                A + (row + 1) * A0 + ks * A1, (row + 1 < M) & (ks < K)
            )
            acc1 += tl.sum(tl.where(valid, a1[None, :] * w, 0.0), 1)
        if BM > 2:
            a2 = _dot_load(
                A + (row + 2) * A0 + ks * A1, (row + 2 < M) & (ks < K)
            )
            acc2 += tl.sum(tl.where(valid, a2[None, :] * w, 0.0), 1)
        if BM > 3:
            a3 = _dot_load(
                A + (row + 3) * A0 + ks * A1, (row + 3 < M) & (ks < K)
            )
            acc3 += tl.sum(tl.where(valid, a3[None, :] * w, 0.0), 1)
        if BM > 4:
            a4 = _dot_load(
                A + (row + 4) * A0 + ks * A1, (row + 4 < M) & (ks < K)
            )
            acc4 += tl.sum(tl.where(valid, a4[None, :] * w, 0.0), 1)
        if BM > 5:
            a5 = _dot_load(
                A + (row + 5) * A0 + ks * A1, (row + 5 < M) & (ks < K)
            )
            acc5 += tl.sum(tl.where(valid, a5[None, :] * w, 0.0), 1)
        if BM > 6:
            a6 = _dot_load(
                A + (row + 6) * A0 + ks * A1, (row + 6 < M) & (ks < K)
            )
            acc6 += tl.sum(tl.where(valid, a6[None, :] * w, 0.0), 1)
        if BM > 7:
            a7 = _dot_load(
                A + (row + 7) * A0 + ks * A1, (row + 7 < M) & (ks < K)
            )
            acc7 += tl.sum(tl.where(valid, a7[None, :] * w, 0.0), 1)
        ks += BK * SPLIT
    if BM > 0:
        tl.store(
            P + part * M * N + (row + 0) * N + cols,
            acc0,
            (row + 0 < M) & (cols < N),
        )
    if BM > 1:
        tl.store(
            P + part * M * N + (row + 1) * N + cols,
            acc1,
            (row + 1 < M) & (cols < N),
        )
    if BM > 2:
        tl.store(
            P + part * M * N + (row + 2) * N + cols,
            acc2,
            (row + 2 < M) & (cols < N),
        )
    if BM > 3:
        tl.store(
            P + part * M * N + (row + 3) * N + cols,
            acc3,
            (row + 3 < M) & (cols < N),
        )
    if BM > 4:
        tl.store(
            P + part * M * N + (row + 4) * N + cols,
            acc4,
            (row + 4 < M) & (cols < N),
        )
    if BM > 5:
        tl.store(
            P + part * M * N + (row + 5) * N + cols,
            acc5,
            (row + 5 < M) & (cols < N),
        )
    if BM > 6:
        tl.store(
            P + part * M * N + (row + 6) * N + cols,
            acc6,
            (row + 6 < M) & (cols < N),
        )
    if BM > 7:
        tl.store(
            P + part * M * N + (row + 7) * N + cols,
            acc7,
            (row + 7 < M) & (cols < N),
        )


@triton.jit
def _sum_splits(
    P,
    Y,
    COUNT: tl.constexpr,
    SPLIT: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for part in tl.static_range(SPLIT):
        total += tl.load(P + part * COUNT + index, index < COUNT, 0)
    tl.store(Y + index, total, index < COUNT)


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    """Decode AWQ weights and reduce products on the Kunlunxin vector cores."""
    m, k = input.shape
    wk, packed_n = qweight.shape
    n = packed_n * 8
    groups = scales.shape[0]
    split = int(split_k_iters)
    if wk != k or groups == 0 or k % groups or scales.shape != (groups, n):
        raise ValueError("weight and scale dimensions must match the input")
    if qzeros.shape != (groups, packed_n) or split < 1:
        raise ValueError(
            "qzeros must match scales, and split_k_iters must be positive"
        )
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("input must be fp16, bf16, or fp32")
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    if m == 0 or n == 0:
        return out
    bm, bn = min(8, triton.next_power_of_2(m)), 16
    bk = min(512, max(128, triton.next_power_of_2(triton.cdiv(k, 16))))
    split = min(split, triton.cdiv(k, bk))
    partial = (
        torch.empty((split, m, n), device=input.device, dtype=torch.float32)
        if split > 1
        else out
    )
    bound = max(
        [max(split * m * n, k * n) + 8192]
        + [
            sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
            for t in (input, qweight, scales, qzeros)
        ]
    )
    weight = torch.empty((n, k), device=input.device, dtype=input.dtype)
    decode_block = min(
        8192, max(1024, triton.next_power_of_2(max(1, k * n // 24)))
    )
    _dequantize[(triton.cdiv(k * n, decode_block),)](
        qweight,
        scales,
        qzeros,
        weight,
        k,
        n,
        k // groups,
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
        bound >= 2**31,
        decode_block,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    _dense[(triton.cdiv(m, bm) * triton.cdiv(n, bn), split)](
        input,
        weight,
        partial,
        m,
        n,
        k,
        split,
        *input.stride(),
        bound >= 2**31,
        bm,
        bn,
        bk,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    if split > 1:
        _sum_splits[(triton.cdiv(m * n, 1024),)](
            partial,
            out,
            m * n,
            split,
            bound >= 2**31,
            1024,
            num_warps=4,
            num_stages=1,
        )
    return out


solution = awq_gemm

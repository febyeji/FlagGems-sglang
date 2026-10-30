# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["mxfp8_block_scaled_matmul"]


@triton.jit
def _decode_e4m3(byte):
    bits = byte.to(tl.uint32)
    magnitude = bits & 127
    exponent, mantissa = magnitude >> 3, magnitude & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(
        tl.float32, bitcast=True
    )
    value = tl.where(
        exponent == 0, mantissa.to(tl.float32) * (1.0 / 512), normal
    )
    value = tl.where(magnitude == 127, float("nan"), value)
    return tl.where((bits & 128) != 0, -value, value)


@triton.jit
def _scale(byte):
    exponent = byte.to(tl.uint32)
    return tl.where(exponent == 0, 0x00400000, exponent << 23).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _decode(
    X,
    S,
    D,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    row, col = index // K, index % K
    valid = index < ROWS * K
    byte = tl.load(X + row * X0 + col * X1, valid, 0)
    exponent = tl.load(S + row * S0 + (col // 32) * S1, valid, 127)
    tl.store(D + index, _decode_e4m3(byte) * _scale(exponent), valid)


@triton.jit
def _load(pointer, mask):
    if pointer.dtype.element_ty == tl.bfloat16:
        word = tl.load(pointer.to(tl.pointer_type(tl.uint16)), mask, other=0)
        return (word.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(pointer, mask, other=0).to(tl.float32)


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
        w = _load(W + cols[:, None] * K + ks[None, :], valid)
        if BM > 0:
            a0 = _load(A + (row + 0) * A0 + ks * A1, (row + 0 < M) & (ks < K))
            acc0 += tl.sum(tl.where(valid, a0[None, :] * w, 0.0), 1)
        if BM > 1:
            a1 = _load(A + (row + 1) * A0 + ks * A1, (row + 1 < M) & (ks < K))
            acc1 += tl.sum(tl.where(valid, a1[None, :] * w, 0.0), 1)
        if BM > 2:
            a2 = _load(A + (row + 2) * A0 + ks * A1, (row + 2 < M) & (ks < K))
            acc2 += tl.sum(tl.where(valid, a2[None, :] * w, 0.0), 1)
        if BM > 3:
            a3 = _load(A + (row + 3) * A0 + ks * A1, (row + 3 < M) & (ks < K))
            acc3 += tl.sum(tl.where(valid, a3[None, :] * w, 0.0), 1)
        if BM > 4:
            a4 = _load(A + (row + 4) * A0 + ks * A1, (row + 4 < M) & (ks < K))
            acc4 += tl.sum(tl.where(valid, a4[None, :] * w, 0.0), 1)
        if BM > 5:
            a5 = _load(A + (row + 5) * A0 + ks * A1, (row + 5 < M) & (ks < K))
            acc5 += tl.sum(tl.where(valid, a5[None, :] * w, 0.0), 1)
        if BM > 6:
            a6 = _load(A + (row + 6) * A0 + ks * A1, (row + 6 < M) & (ks < K))
            acc6 += tl.sum(tl.where(valid, a6[None, :] * w, 0.0), 1)
        if BM > 7:
            a7 = _load(A + (row + 7) * A0 + ks * A1, (row + 7 < M) & (ks < K))
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


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    """Compute scaled E4M3 products using Kunlunxin vector reductions."""
    m, k = a.shape
    n, bk = b.shape
    if bk != k or k == 0 or k % 32:
        raise ValueError("K must agree and be a positive multiple of 32")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("a and b must be E4M3FN tensors")
    if a_scale.shape != (m, k // 32) or b_scale.shape != (n, k // 32):
        raise ValueError("scales must contain one exponent per 32 elements")
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise ValueError("scales must use the uint8 UE8M0 representation")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("output_dtype must be fp16, bf16, or fp32")
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    bound = max(
        [max(m * n, m * k, n * k) + 4096]
        + [
            sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
            for t in (a, a_scale, b, b_scale)
        ]
    )
    bm = min(8, triton.next_power_of_2(m))
    bk = min(512, max(128, triton.next_power_of_2(triton.cdiv(k, 16))))
    ad = torch.empty((m, k), device=a.device, dtype=torch.float32)
    bd = torch.empty((n, k), device=b.device, dtype=torch.float32)
    for x, scale, decoded, rows in [(a, a_scale, ad, m), (b, b_scale, bd, n)]:
        _decode[(triton.cdiv(rows * k, 1024),)](
            x.view(torch.uint8),
            scale,
            decoded,
            rows,
            k,
            *x.stride(),
            *scale.stride(),
            bound >= 2**31,
            1024,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    _dense[(triton.cdiv(m, bm) * triton.cdiv(n, 8), 1)](
        ad,
        bd,
        out,
        m,
        n,
        k,
        1,
        k,
        1,
        bound >= 2**31,
        bm,
        8,
        bk,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return out


solution = mxfp8_block_scaled_matmul

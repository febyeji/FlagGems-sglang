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
def _decode_rows(
    X,
    S,
    D,
    F,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    WIDE: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    if WIDE:
        row, part = row.to(tl.int64), part.to(tl.int64)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(X + row * X0 + cols * X1, cols < K, 0)
    exponent = tl.load(S + row * S0 + (cols // 32) * S1, cols < K, 127)
    value = _decode_e4m3(byte) * _scale(exponent)
    tl.store(D + row * K + cols, value, cols < K)
    if tl.program_id(1) == 0:
        group = tl.arange(0, REDUCE)
        e = tl.load(S + row * S0 + tl.minimum(group, K // 32 - 1) * S1)
        unsafe = tl.max(((e < 10) | (e > 246)).to(tl.int32), 0)
        tl.store(F + row, unsafe)


@triton.jit
def _matmul(
    A,
    AF,
    B,
    BF,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    if WIDE:
        tile = tile.to(tl.int64)
    rows = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    cols = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    af = tl.load(AF + tl.minimum(rows, M - 1))
    bf = tl.load(BF + tl.minimum(cols, N - 1))
    safe = (tl.max(af, 0) | tl.max(bf, 0)) == 0
    ap = A + rows[:, None] * K + ks[None, :]
    bp = B + cols[None, :] * K + ks[:, None]
    acc = tl.full((BM, BN), 0, tl.float32)
    if safe:
        for step in range(tl.cdiv(K, BK)):
            a = tl.load(
                ap, (rows[:, None] < M) & (ks[None, :] < K - step * BK), 0
            )
            b = tl.load(
                bp, (cols[None, :] < N) & (ks[:, None] < K - step * BK), 0
            )
            acc = tl.dot(
                a.to(tl.bfloat16),
                b.to(tl.bfloat16),
                acc,
                input_precision="ieee",
            )
            ap += BK
            bp += BK
    else:
        for step in range(tl.cdiv(K, BK)):
            a = tl.load(
                ap, (rows[:, None] < M) & (ks[None, :] < K - step * BK), 0
            )
            b = tl.load(
                bp, (cols[None, :] < N) & (ks[:, None] < K - step * BK), 0
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")
            ap += BK
            bp += BK
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    """Decode MXFP8 once and multiply with exact BF16 or FP32 operands."""
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
    bm = 16 if m <= 16 else (32 if m < 128 else 64)
    bn = 128 if m >= 512 and n >= 1024 else 64
    block_k = 64 if m < 128 else 32
    bound = max(
        [max(m * n, m * k, n * k) + 4096]
        + [
            sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
            for t in (a, a_scale, b, b_scale)
        ]
    )
    wide = bound >= 2**31
    ad = torch.empty((m, k), device=a.device, dtype=torch.float32)
    bd = torch.empty((n, k), device=b.device, dtype=torch.float32)
    af = torch.empty((m,), device=a.device, dtype=torch.int32)
    bf = torch.empty((n,), device=b.device, dtype=torch.int32)
    for x, scale, decoded, flags, rows in [
        (a, a_scale, ad, af, m),
        (b, b_scale, bd, bf, n),
    ]:
        _decode_rows[(rows, triton.cdiv(k, 1024))](
            x.view(torch.uint8),
            scale,
            decoded,
            flags,
            rows,
            k,
            *x.stride(),
            *scale.stride(),
            wide,
            triton.next_power_of_2(k // 32),
            1024,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    _matmul[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        ad,
        af,
        bd,
        bf,
        out,
        m,
        n,
        k,
        wide,
        bm,
        bn,
        block_k,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return out


solution = mxfp8_block_scaled_matmul

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["mxfp8_block_scaled_matmul"]


@triton.jit
def _decode_e4m3(byte):
    bits = byte.to(tl.int32)
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
    exponent = byte.to(tl.int32)
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
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    lane = tl.arange(0, BLOCK)
    for step in range(triton.cdiv(ROWS * K, PROGRAMS * BLOCK)):
        index = (step * PROGRAMS + pid) * BLOCK + lane
        row, col = index // K, index % K
        valid = index < ROWS * K
        byte = tl.load(X + row * X0 + col * X1, valid, 0)
        exponent = tl.load(S + row * S0 + (col // 32) * S1, valid, 127)
        tl.store(D + index, _decode_e4m3(byte) * _scale(exponent), valid)


@triton.jit
def _matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    tiles: tl.constexpr = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    for step in range(triton.cdiv(tiles, PROGRAMS)):
        tile = step * PROGRAMS + pid
        rows = tile // triton.cdiv(N, BN) * BM + tl.arange(0, BM)
        cols = tile % triton.cdiv(N, BN) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(triton.cdiv(K, BK)):
            offsets = block * BK + ks
            a = tl.load(
                A + rows[:, None] * K + offsets[None, :],
                (rows[:, None] < M) & (offsets[None, :] < K),
                0,
            )
            b = tl.load(
                B + cols[None, :] * K + offsets[:, None],
                (cols[None, :] < N) & (offsets[:, None] < K),
                0,
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < M) & (cols[None, :] < N),
        )


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    """Decode MXFP8 to FP32 and use a bounded Ascend matrix launch."""
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
    bm, bn, block_k = 32, 64, 64
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
    for x, scale, decoded, rows in [(a, a_scale, ad, m), (b, b_scale, bd, n)]:
        programs = min(40, triton.cdiv(rows * k, 1024))
        _decode[(programs,)](
            x.view(torch.uint8),
            scale,
            decoded,
            rows,
            k,
            *x.stride(),
            *scale.stride(),
            wide,
            programs,
            1024,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    programs = min(20, triton.cdiv(m, bm) * triton.cdiv(n, bn))
    _matmul[(programs,)](
        ad,
        bd,
        out,
        m,
        n,
        k,
        wide,
        programs,
        bm,
        bn,
        block_k,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


solution = mxfp8_block_scaled_matmul

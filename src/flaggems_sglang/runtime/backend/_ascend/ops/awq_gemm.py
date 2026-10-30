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
def _gemm(
    A,
    W,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    SPLIT: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    Z0: tl.constexpr,
    Z1: tl.constexpr,
    WIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    tiles: tl.constexpr = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    for task in range(pid, tiles * SPLIT, PROGRAMS):
        part, tile = task // tiles, task % tiles
        rows = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
        cols = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
        ks = part * BK + tl.arange(0, BK)
        packed_cols = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(
            0, BN // 8
        )
        nibble = tl.arange(0, 8)
        shifts = ((nibble // 2) + (nibble % 2) * 4) * 4
        acc = tl.full((BM, BN), 0, tl.float32)
        for _ in range(tl.cdiv(K, BK * SPLIT)):
            a = _load(
                A + rows[:, None] * A0 + ks[None, :] * A1,
                (rows[:, None] < M) & (ks[None, :] < K),
            )
            packed_valid = (ks[:, None] < K) & (packed_cols[None, :] < N // 8)
            words = tl.load(
                W + ks[:, None] * W0 + packed_cols[None, :] * W1,
                packed_valid,
                0,
            )
            w = tl.reshape(
                (words[:, :, None] >> shifts[None, None, :]) & 15, (BK, BN)
            )
            if GROUP % BK == 0:
                group = (part * BK + _ * BK * SPLIT) // GROUP
                z_words = tl.load(
                    Z + group * Z0 + packed_cols * Z1,
                    (group < tl.cdiv(K, GROUP)) & (packed_cols < N // 8),
                    0,
                )
                z = tl.reshape(
                    (z_words[:, None] >> shifts[None, :]) & 15, (BN,)
                )
                scale = _load(
                    S + group * S0 + cols * S1,
                    (group < tl.cdiv(K, GROUP)) & (cols < N),
                )
                weight = (w - z[None, :]).to(tl.float32) * scale[None, :]
            else:
                z_words = tl.load(
                    Z
                    + (ks[:, None] // GROUP) * Z0
                    + packed_cols[None, :] * Z1,
                    packed_valid,
                    0,
                )
                z = tl.reshape(
                    (z_words[:, :, None] >> shifts[None, None, :]) & 15,
                    (BK, BN),
                )
                scale = _load(
                    S + (ks[:, None] // GROUP) * S0 + cols[None, :] * S1,
                    (ks[:, None] < K) & (cols[None, :] < N),
                )
                weight = (w - z).to(tl.float32) * scale
            weight = weight.to(S.dtype.element_ty).to(A.dtype.element_ty)
            acc = tl.dot(
                a.to(A.dtype.element_ty),
                weight,
                acc,
                input_precision=PRECISION,
            )
            ks += BK * SPLIT
        index = part * M * N + rows[:, None] * N + cols[None, :]
        tl.store(P + index, acc, (rows[:, None] < M) & (cols[None, :] < N))


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
    for part in range(SPLIT):
        total += tl.load(P + part * COUNT + index, index < COUNT, 0)
    tl.store(Y + index, total, index < COUNT)


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    """Multiply by AWQ weights with at most 20 fused Ascend programs."""
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
    bm = 16 if m <= 16 else (32 if m < 128 else 64)
    bn, bk = 64, (64 if m < 128 else 32)
    split = min(split, triton.cdiv(k, bk))
    partial = (
        torch.empty((split, m, n), device=input.device, dtype=torch.float32)
        if split > 1
        else out
    )
    bound = max(
        [split * m * n + 4096]
        + [
            sum((s - 1) * d for s, d in zip(t.shape, t.stride()))
            for t in (input, qweight, scales, qzeros)
        ]
    )
    precision = "ieee"
    if input.dtype == torch.float32:
        precision = "hf32" if input.device.type == "npu" else "tf32"
    programs = min(20, triton.cdiv(m, bm) * triton.cdiv(n, bn) * split)
    _gemm[(programs,)](
        input,
        qweight,
        scales,
        qzeros,
        partial,
        m,
        n,
        k,
        k // groups,
        split,
        *input.stride(),
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
        bound >= 2**31,
        programs,
        bm,
        bn,
        bk,
        precision,
        num_warps=4,
        num_stages=3,
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

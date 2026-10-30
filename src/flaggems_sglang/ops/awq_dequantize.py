# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["awq_dequantize"]


@triton.jit
def _dequantize_group(
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
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    column_tile = pid % tl.cdiv(N, COLS)
    row_tile = pid // tl.cdiv(N, COLS)
    group = row_tile // tl.cdiv(GROUP, ROWS)
    local_row = row_tile % tl.cdiv(GROUP, ROWS) * ROWS + tl.arange(0, ROWS)
    row = group * GROUP + local_row
    col = column_tile * COLS + tl.arange(0, COLS)
    packed_col = column_tile * (COLS // 8) + tl.arange(0, COLS // 8)
    nibble = tl.arange(0, 8)
    shift = (nibble // 2 + nibble % 2 * 4) * 4
    # Reuse each group's zero points and scales across the rows in this tile.
    zero = tl.load(Z + group * Z0 + packed_col * Z1, packed_col < N // 8, 0)
    zero = tl.reshape((zero[:, None] >> shift[None, :]) & 15, (COLS,))
    scale = tl.load(S + group * S0 + col * S1, col < N, 0).to(tl.float32)
    word = tl.load(
        W + row[:, None] * W0 + packed_col[None, :] * W1,
        (local_row[:, None] < GROUP) & (packed_col[None, :] < N // 8),
        0,
    )
    weight = tl.reshape(
        (word[:, :, None] >> shift[None, None, :]) & 15, (ROWS, COLS)
    )
    value = (weight - zero[None, :]).to(tl.float32) * scale[None, :]
    tl.store(
        Y + row[:, None] * N + col[None, :],
        value,
        (local_row[:, None] < GROUP) & (col[None, :] < N),
    )


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
    row, col = index // N, index % N
    shift = ((col % 8) // 2 + (col % 2) * 4) * 4
    word = tl.load(W + row * W0 + (col // 8) * W1, index < K * N, 0)
    zero = tl.load(Z + (row // GROUP) * Z0 + (col // 8) * Z1, index < K * N, 0)
    scale = tl.load(S + (row // GROUP) * S0 + col * S1, index < K * N, 0).to(
        tl.float32
    )
    value = (((word >> shift) & 15) - ((zero >> shift) & 15)).to(
        tl.float32
    ) * scale
    tl.store(Y + index, value, index < K * N)


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
        if k * n > 32768 and k // groups >= 4:
            rows, cols = (8, 128) if k * n < 1048576 else (16, 256)
            tiles = (
                groups * triton.cdiv(k // groups, rows) * triton.cdiv(n, cols)
            )
            _dequantize_group[(tiles,)](
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
                rows,
                cols,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return out
        block = 256 if k * n <= 32768 else 1024
        _dequantize[(triton.cdiv(k * n, block),)](
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

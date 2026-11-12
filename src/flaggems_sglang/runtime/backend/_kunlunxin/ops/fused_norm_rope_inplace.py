# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Kunlunxin head/tail split: 576-wide rows take 576 lanes, not 1024.

The generic kernel covers the row with one [pairs, 2] tile whose width is
the next power of two of the row (1024 lanes for 576 columns). Here the
nope head and the interleaved rope tail are separate exact tiles, as in
the ascend kernel.
"""

import torch
import triton
import triton.language as tl

__all__ = ["fused_norm_rope_inplace"]


@triton.jit
def _split_rows(
    KV,
    W,
    F,
    P,
    OUT,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    FS: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_P: tl.constexpr,
    BH: tl.constexpr,
    HP: tl.constexpr,
    ROWS: tl.constexpr,
):
    N: tl.constexpr = D - R
    EXACT: tl.constexpr = (BH == N) and (M % ROWS == 0)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    head = tl.arange(0, BH)
    pair = tl.arange(0, HP)
    part = tl.arange(0, 2)
    lane = pair[None, :, None] * 2 + part[None, None, :]
    if EXACT:
        live_head = None
        x_head = tl.load(KV + row[:, None] * S + head[None, :]).to(tl.float32)
        x_tail = tl.load(
            KV + row[:, None, None] * S + N + lane
        ).to(tl.float32)
    else:
        live = row[:, None] < M
        live_head = live & (head[None, :] < N)
        live_tail = live[:, :, None] & (lane < R)
        x_head = tl.load(
            KV + row[:, None] * S + head[None, :], live_head, other=0.0
        ).to(tl.float32)
        x_tail = tl.load(
            KV + row[:, None, None] * S + N + lane, live_tail, other=0.0
        ).to(tl.float32)
    squares = tl.sum(x_head * x_head, 1) + tl.sum(tl.sum(x_tail * x_tail, 2), 1)
    inv = tl.rsqrt(squares / D + EPS)
    y_head = x_head * inv[:, None]
    y_tail = x_tail * inv[:, None, None]
    if HAS_W:
        if EXACT:
            y_head = y_head * tl.load(W + head).to(tl.float32)
            y_tail = y_tail * tl.load(W + N + lane).to(tl.float32)
        else:
            y_head = y_head * tl.load(
                W + head, head < N, other=0.0
            ).to(tl.float32)
            y_tail = y_tail * tl.load(
                W + N + lane, lane < R, other=0.0
            ).to(tl.float32)
    if HAS_P:
        if EXACT:
            position = tl.load(P + row).to(tl.int64)
        else:
            position = tl.load(P + row, row < M, other=0).to(tl.int64)
    else:
        position = row
    # The table is interleaved like the tail: (real, imag) per pair. The
    # halves of each pair are picked with a 0/1 selector product summed
    # over the pair axis, which is exact and needs no broadcast `where`.
    rotated = lane < R
    f = tl.load(
        F + position[:, None, None] * FS + lane,
        rotated if EXACT else live_tail,
        other=0.0,
    )
    take_real = (part == 0).to(tl.float32)[None, None, :]
    take_imag = (part == 1).to(tl.float32)[None, None, :]
    f_real = tl.sum(f * take_real, 2)
    f_imag = tl.sum(f * take_imag, 2)
    y_real = tl.sum(y_tail * take_real, 2)
    y_imag = tl.sum(y_tail * take_imag, 2)
    out_tail = (y_real * f_real - y_imag * f_imag)[
        :, :, None
    ] * take_real + (y_real * f_imag + y_imag * f_real)[:, :, None] * take_imag
    if EXACT:
        tl.store(
            OUT + row[:, None] * D + head[None, :],
            y_head.to(OUT.dtype.element_ty),
        )
        tl.store(
            OUT + row[:, None, None] * D + N + lane,
            out_tail.to(OUT.dtype.element_ty),
        )
    else:
        tl.store(
            OUT + row[:, None] * D + head[None, :],
            y_head.to(OUT.dtype.element_ty),
            live_head,
        )
        tl.store(
            OUT + row[:, None, None] * D + N + lane,
            out_tail.to(OUT.dtype.element_ty),
            live_tail,
        )


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    """RMSNorm over the row, then RoPE on its interleaved rope tail."""
    m, d = kv.shape
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if positions is not None and not positions.is_contiguous():
        positions = positions.contiguous()
    freqs = (
        freqs_cis if freqs_cis.stride(-1) == 1 else freqs_cis.contiguous()
    ).view(torch.float32)
    out = torch.empty_like(kv)
    if m:
        r = freqs_cis.shape[-1] * 2
        n = d - r
        bh = 1 << (n - 1).bit_length()
        hp = 1 << max((r // 2) - 1, 0).bit_length()
        rows = 1 << min(3, max(0, (1024 // (bh + 2 * hp)).bit_length() - 1))
        _split_rows.run(
            kv,
            weight,
            freqs,
            positions,
            out,
            m,
            d,
            r,
            kv.stride(0),
            freqs.stride(0),
            eps,
            weight is not None,
            positions is not None,
            bh,
            hp,
            rows,
            grid=((m + rows - 1) // rows,),
            warmup=False,
            num_warps=1,
            isCloseMemoryAsync=False,
        )
    return out


solution = fused_norm_rope_inplace

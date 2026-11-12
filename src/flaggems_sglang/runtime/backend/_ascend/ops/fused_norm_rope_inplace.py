# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_norm_rope_inplace"]

# One program per vector core: the SGLang kernel launches one program per
# row, and on this chip each program is dispatched on its own, so the rows
# are walked by a fixed set of programs instead.
CORES = 40


@triton.jit(
    do_not_specialize=["KV", "W", "F", "P", "OUT"],
    do_not_specialize_on_alignment=["KV", "W", "F", "P", "OUT"],
)
def _blocks(
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
    GRID: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    # The trip count is fixed at compile time and each pass takes ROWS rows
    # as one tile, the shape the merged RMSNorm+RoPE kernels use on this
    # chip: 2-D loads, one reduction along the row axis, 2-D stores.
    #
    # The row is two contiguous blocks, the part before the rope tail and
    # the tail as [rows, pairs, 2], each read once and written once. The
    # halves of each pair are taken apart and put back together in
    # registers: a product with a 0/1 selector summed over the pair is
    # exact, and every access stays a contiguous block.
    N: tl.constexpr = D - R
    head = tl.arange(0, BH)
    pair = tl.arange(0, HP)
    part = tl.arange(0, 2)
    lane = pair[:, None] * 2 + part[None, :]
    rows_in_tile = tl.arange(0, ROWS)
    take_real = (part == 0).to(tl.float32)[None, :]
    take_imag = (part == 1).to(tl.float32)[None, :]
    if HAS_W:
        if N > 0:
            w_head = tl.load(W + head, head < N, other=0.0).to(tl.float32)
        w_tail = tl.load(W + N + lane, lane < R, other=0.0).to(tl.float32)
    for chunk in range(CHUNKS):
        row = (
            tl.program_id(0) + chunk * GRID
        ).to(tl.int64) * ROWS + rows_in_tile.to(tl.int64)
        live = row < M
        head_mask = live[:, None] & (head < N)[None, :]
        tail_mask = live[:, None, None] & (lane < R)[None, :, :]
        if HAS_P:
            position = tl.load(P + row, live, other=0).to(tl.int64)
        else:
            position = row
        x_tail = tl.load(
            KV + row[:, None, None] * S + (N + lane)[None, :, :],
            tail_mask,
            other=0.0,
        ).to(tl.float32)
        squares = tl.sum(tl.sum(x_tail * x_tail, 2), 1)
        if N > 0:
            x_head = tl.load(
                KV + row[:, None] * S + head[None, :], head_mask, other=0.0
            ).to(tl.float32)
            squares += tl.sum(x_head * x_head, 1)
        inv = tl.rsqrt(squares / D + EPS)
        if N > 0:
            y_head = x_head * inv[:, None]
            if HAS_W:
                y_head = y_head * w_head[None, :]
            tl.store(
                OUT + row[:, None] * D + head[None, :],
                y_head.to(OUT.dtype.element_ty),
                head_mask,
            )
        y_tail = x_tail * inv[:, None, None]
        if HAS_W:
            y_tail = y_tail * w_tail[None, :, :]
        y_real = tl.sum(y_tail * take_real[None, :, :], 2)
        y_imag = tl.sum(y_tail * take_imag[None, :, :], 2)
        # The table is interleaved like the row: (real, imag) per pair.
        f = tl.load(
            F + position[:, None, None] * FS + lane[None, :, :],
            tail_mask,
            other=0.0,
        )
        f_real = tl.sum(f * take_real[None, :, :], 2)
        f_imag = tl.sum(f * take_imag[None, :, :], 2)
        out_real = y_real * f_real - y_imag * f_imag
        out_imag = y_real * f_imag + y_imag * f_real
        out = (
            out_real[:, :, None] * take_real[None, :, :]
            + out_imag[:, :, None] * take_imag[None, :, :]
        )
        tl.store(
            OUT + row[:, None, None] * D + (N + lane)[None, :, :],
            out.to(OUT.dtype.element_ty),
            tail_mask,
        )


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    """RMSNorm over the row, then RoPE on its interleaved rope tail.

    The SGLang kernel rewrites ``kv`` in place; this writes a new tensor and
    leaves ``kv`` as it was, which is what the task's reference returns.
    """
    m, d = kv.shape
    # The task promises only that kv's last dimension is dense. The kernel
    # reads weight and positions at unit stride, so a strided view of
    # either is made contiguous first; is_contiguous() is a flag read and
    # costs half a microsecond when there is nothing to do.
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if positions is not None and not positions.is_contiguous():
        positions = positions.contiguous()
    # The kernel indexes the table as interleaved (real, imag) floats, which
    # needs the complex entries of a row next to each other.
    freqs = (
        freqs_cis if freqs_cis.stride(-1) == 1 else freqs_cis.contiguous()
    ).view(torch.float32)
    out = torch.empty_like(kv)
    if m:
        half = freqs_cis.shape[-1]
        r = 2 * half
        bh = 1 << max(d - r - 1, 0).bit_length()
        hp = 1 << max(half - 1, 0).bit_length()
        # Batch rows into one tile once there are more rows than cores, so
        # every pass moves a full 2-D block instead of one row at a time.
        rows = 1
        while rows < 8 and rows * CORES * 2 <= m:
            rows *= 2
        grid = min((m + rows - 1) // rows, CORES)
        _blocks.run(
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
            grid,
            ((m + rows - 1) // rows + grid - 1) // grid,
            grid=(grid,),
            warmup=False,
            num_warps=4,
        )
    return out


solution = fused_norm_rope_inplace

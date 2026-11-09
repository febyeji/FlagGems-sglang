# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["apply_rotary_emb"]

# One program per vector core: the SGLang kernel launches one program per
# (token, head), and on this chip each program is dispatched on its own, so
# the rows are walked by a fixed set of programs instead.
CORES = 40


@triton.jit(
    do_not_specialize=["X", "F", "P", "OUT"],
    do_not_specialize_on_alignment=["X", "F", "P", "OUT"],
)
def _rows(
    X,
    F,
    P,
    OUT,
    BLOCKS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    FS: tl.constexpr,
    PS: tl.constexpr,
    HAS_P: tl.constexpr,
    INVERSE: tl.constexpr,
    SHARED: tl.constexpr,
    PAIRS: tl.constexpr,
    ROWS: tl.constexpr,
    GRID: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    # The trip count is fixed at compile time and each pass takes one block
    # of rows with a scalar index, which is the shape this chip's compiler
    # takes. A row is 64 elements on this operator, a tenth of what a pass
    # moved on the batch's other rope kernel, so a pass takes several rows
    # at once: rows are adjacent in memory, so the block is still one
    # contiguous [rows * pairs, 2] tile, read once and written once.
    #
    # The halves of each pair are taken apart and put back together in
    # registers, a product with a 0/1 selector summed over the pair:
    # stride-2 stores into a row left it wrong here, and reads shifted by
    # one element are misaligned.
    line = tl.arange(0, ROWS * PAIRS)
    part = tl.arange(0, 2)
    lane = line[:, None] * 2 + part[None, :]
    take_real = (part == 0).to(tl.float32)[None, :]
    take_imag = (part == 1).to(tl.float32)[None, :]
    for chunk in range(CHUNKS):
        block = (tl.program_id(0) + chunk * GRID).to(tl.int64)
        row = block * ROWS
        if HEADS == 1:
            token = row
        else:
            token = row // HEADS
        # The blocks divide among the programs when their count is a
        # multiple of GRID, and then no access needs a block mask.
        if BLOCKS % GRID == 0:
            mask = lane < ROWS * D
            if HAS_P:
                position = tl.load(P + token * PS).to(tl.int64)
            else:
                position = token
        else:
            keep = block < BLOCKS
            mask = (lane < ROWS * D) & keep
            if HAS_P:
                position = tl.load(P + token * PS, keep, other=0).to(tl.int64)
            else:
                position = token
        x = tl.load(
            X + token * SX0 + (row - token * HEADS) * SX1 + lane,
            mask,
            other=0.0,
        ).to(tl.float32)
        x_real = tl.sum(x * take_real, 1)
        x_imag = tl.sum(x * take_imag, 1)
        # The table is interleaved like the row: (real, imag) per pair.
        # Every head of a token takes the same row, so a block inside one
        # token reads it once and repeats it; without positions the rows
        # of a block are adjacent in the table and come in as one block.
        if SHARED:
            pair = tl.arange(0, PAIRS)
            narrow = pair[:, None] * 2 + part[None, :]
            fmask = narrow < D
            if BLOCKS % GRID != 0:
                # A trailing pass sits on blocks past the table's rows, so
                # the read needs the block mask too.
                fmask = fmask & keep
            f = tl.load(F + position * FS + narrow, fmask, other=0.0)
            if ROWS == 1:
                f_real = tl.sum(f * take_real, 1)
                f_imag = tl.sum(f * take_imag, 1)
            else:
                f_real = tl.broadcast_to(
                    tl.sum(f * take_real, 1)[None, :], (ROWS, PAIRS)
                ).reshape(ROWS * PAIRS)
                f_imag = tl.broadcast_to(
                    tl.sum(f * take_imag, 1)[None, :], (ROWS, PAIRS)
                ).reshape(ROWS * PAIRS)
        else:
            f = tl.load(F + row * FS + lane, mask, other=0.0)
            f_real = tl.sum(f * take_real, 1)
            f_imag = tl.sum(f * take_imag, 1)
        if INVERSE:
            out_real = x_real * f_real + x_imag * f_imag
            out_imag = x_imag * f_real - x_real * f_imag
        else:
            out_real = x_real * f_real - x_imag * f_imag
            out_imag = x_real * f_imag + x_imag * f_real
        out = out_real[:, None] * take_real + out_imag[:, None] * take_imag
        tl.store(OUT + row * D + lane, out.to(OUT.dtype.element_ty), mask)


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    """Interleaved-complex RoPE over the last dim of ``x``.

    The SGLang function rewrites ``x`` in place and returns it; this writes
    a new tensor and leaves ``x`` as it was, which is what the task's
    reference returns.
    """
    heads = x.shape[1] if x.ndim == 3 else 1
    d = x.shape[-1]
    n = x.shape[0] * heads
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if n:
        # Both tables are indexed as interleaved (real, imag) floats, which
        # needs the pair halves of a row next to each other.
        if x.stride(-1) != 1:
            x = x.contiguous()
        freqs = (
            freqs_cis if freqs_cis.stride(-1) == 1 else freqs_cis.contiguous()
        ).view(torch.float32)
        pairs = 1 << ((d + 1) // 2 - 1).bit_length()
        # A block of rows is one tile only where its rows are adjacent in
        # x and share one frequency row, or follow each other in the table.
        flat = x.stride(0) == heads * d and (
            x.ndim == 2 or x.stride(1) == d
        )
        rows = 8 if flat and 2 * pairs == d else 1
        while rows > 1 and not (
            n % rows == 0
            and (
                heads % rows == 0
                or (heads == 1 and positions is None and freqs.stride(0) == d)
            )
        ):
            rows //= 2
        blocks = n // rows
        grid = min(blocks, CORES)
        _rows.run(
            x,
            freqs,
            positions,
            out,
            blocks,
            heads,
            d,
            x.stride(0),
            x.stride(1) if x.ndim == 3 else 0,
            freqs.stride(0),
            positions.stride(0) if positions is not None else 0,
            positions is not None,
            inverse,
            rows == 1 or heads % rows == 0,
            pairs,
            rows,
            grid,
            (blocks + grid - 1) // grid,
            grid=(grid,),
            warmup=False,
        )
    return out


solution = apply_rotary_emb

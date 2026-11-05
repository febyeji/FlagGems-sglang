# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["apply_rotary_emb"]

# Card B took four warps per program where the other chips took
# one on the batch's other rope kernel, so its copy keeps four.


@triton.jit
def _rotate(
    X,
    F,
    P,
    OUT,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    FS: tl.constexpr,
    HAS_P: tl.constexpr,
    INVERSE: tl.constexpr,
    FLAT: tl.constexpr,
    PAIRS: tl.constexpr,
    ROWS: tl.constexpr,
):
    # A row is read as [pairs, 2], one (real, imag) pair to a line, so the
    # interleaved layout needs no strided access: the pair halves are taken
    # apart in registers and the row leaves in one contiguous store. Rows
    # are the flattened (token, head) pairs, which keeps the 2-D and 3-D
    # cases the same kernel; every head of a token reads the same frequency
    # row, and consecutive rows of a program are consecutive heads.
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    pair = tl.arange(0, PAIRS)
    part = tl.arange(0, 2)
    column = pair[None, :, None] * 2 + part[None, None, :]
    if HEADS == 1:
        token = row
    else:
        token = row // HEADS
    if FLAT:
        base = row * D
    else:
        base = token * SX0 + (row - token * HEADS) * SX1
    # N is known at compile time, so a count that divides into whole
    # programs needs no row mask, and a width that fills the tile needs no
    # column mask.
    if N % ROWS == 0:
        if D == 2 * PAIRS:
            x = tl.load(X + base[:, None, None] + column).to(tl.float32)
        else:
            x = tl.load(
                X + base[:, None, None] + column, column < D, other=0.0
            ).to(tl.float32)
        if HAS_P:
            position = tl.load(P + token).to(tl.int64)
        else:
            position = token
    else:
        live = row < N
        if D == 2 * PAIRS:
            inside = live[:, None, None]
        else:
            inside = live[:, None, None] & (column < D)
        x = tl.load(X + base[:, None, None] + column, inside, other=0.0).to(
            tl.float32
        )
        if HAS_P:
            position = tl.load(P + token, live, other=0).to(tl.int64)
        else:
            position = token
    # The table is interleaved like the row: (real, imag) per pair.
    source = F + position[:, None, None] * FS + column
    if D == 2 * PAIRS:
        f = tl.load(source)
    else:
        f = tl.load(source, column < D, other=0.0)
    real = part[None, None, :] == 0
    x_real = tl.sum(tl.where(real, x, 0.0), 2)
    x_imag = tl.sum(tl.where(real, 0.0, x), 2)
    f_real = tl.sum(tl.where(real, f, 0.0), 2)
    f_imag = tl.sum(tl.where(real, 0.0, f), 2)
    if INVERSE:
        out_real = x_real * f_real + x_imag * f_imag
        out_imag = x_imag * f_real - x_real * f_imag
    else:
        out_real = x_real * f_real - x_imag * f_imag
        out_imag = x_real * f_imag + x_imag * f_real
    out = tl.where(real, out_real[:, :, None], out_imag[:, :, None]).to(
        OUT.dtype.element_ty
    )
    if N % ROWS == 0:
        if D == 2 * PAIRS:
            tl.store(OUT + row[:, None, None] * D + column, out)
        else:
            tl.store(OUT + row[:, None, None] * D + column, out, column < D)
    else:
        tl.store(OUT + row[:, None, None] * D + column, out, inside)


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    """Interleaved-complex RoPE over the last dim of ``x``.

    The SGLang function rewrites ``x`` in place and returns it; this writes
    a new tensor and leaves ``x`` as it was, which is what the task's
    reference returns.
    """
    # The host side is most of a small call, so it stays in plain Python:
    # triton.next_power_of_2 and triton.cdiv are JIT functions in Triton 3.9
    # and cost about 9 us each, and a complex64 table viewed as float32 is
    # a cheaper view than view_as_real.
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
        rows = max(1, min(8, 1024 // (2 * pairs)))
        while n % rows:
            rows //= 2
        stride_head = x.stride(1) if x.ndim == 3 else 0
        _rotate.run(
            x,
            freqs,
            positions,
            out,
            n,
            heads,
            d,
            x.stride(0),
            stride_head,
            freqs.stride(0),
            positions is not None,
            inverse,
            x.stride(0) == heads * d and stride_head in (0, d),
            pairs,
            rows,
            grid=((n + rows - 1) // rows,),
            warmup=False,
            num_warps=4,
        )
    return out


solution = apply_rotary_emb

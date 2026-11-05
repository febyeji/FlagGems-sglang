# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["apply_rotary_emb"]


@triton.jit
def _row(
    X,
    F,
    P,
    OUT,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    FS: tl.constexpr,
    HAS_P: tl.constexpr,
    INVERSE: tl.constexpr,
    B: tl.constexpr,
):
    # One program per (token, head) row, as the SGLang kernel runs, and
    # every access is a one-dimensional contiguous block: the row read
    # once, the same row one element to each side for the other half of
    # each pair, and one store. Reading the row as a [pairs, 2] block
    # instead ended in a launch failure on this chip from the fourth case
    # on.
    #
    # The tile is twice the width of the row, so the row's last element is
    # not the tile's last lane. At equal widths this chip returned a wrong
    # value in the last column of every row, the same value whether the
    # two shifted reads were blended by a select or by a sum of products,
    # so the lane itself is what it mishandles; the batch's other rope
    # kernel passes here with the row inside a wider tile.
    row = tl.program_id(0).to(tl.int64)
    if HEADS == 1:
        token = row
    else:
        token = row // HEADS
    column = tl.arange(0, B)
    inside = column < D
    odd = (column & 1).to(tl.float32)
    even = 1.0 - odd
    to_next = column < D - 1
    to_prev = (column >= 1) & inside
    source = X + token * SX0 + (row - token * HEADS) * SX1 + column
    x = tl.load(source, inside, other=0.0).to(tl.float32)
    prior = tl.load(source - 1, to_prev, other=0.0).to(tl.float32)
    following = tl.load(source + 1, to_next, other=0.0).to(tl.float32)
    partner = prior * odd + following * even
    if HAS_P:
        position = tl.load(P + token).to(tl.int64)
    else:
        position = token
    # The table is interleaved like the row, so element c of the row reads
    # entry c: the real part on even c, the imaginary part on odd c.
    freq = F + position * FS + column
    f_same = tl.load(freq, inside, other=0.0)
    f_next = tl.load(freq + 1, to_next, other=0.0)
    f_prev = tl.load(freq - 1, to_prev, other=0.0)
    if INVERSE:
        # Even c: real * f_real + imag * f_imag.
        # Odd c: imag * f_real - real * f_imag.
        out = (x * f_prev - partner * f_same) * odd + (
            x * f_same + partner * f_next
        ) * even
    else:
        # Even c: real * f_real - imag * f_imag.
        # Odd c: imag * f_real + real * f_imag.
        out = (x * f_prev + partner * f_same) * odd + (
            x * f_same - partner * f_next
        ) * even
    tl.store(OUT + row * D + column, out.to(OUT.dtype.element_ty), inside)


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
        # One lane of slack past the row at least: see the kernel.
        b = 1 << d.bit_length()
        _row.run(
            x,
            freqs,
            positions,
            out,
            heads,
            d,
            x.stride(0),
            x.stride(1) if x.ndim == 3 else 0,
            freqs.stride(0),
            positions is not None,
            inverse,
            b,
            grid=(n,),
            warmup=False,
            num_warps=4 if b <= 4096 else 8,
            isCloseMemoryAsync=False,
        )
    return out


solution = apply_rotary_emb

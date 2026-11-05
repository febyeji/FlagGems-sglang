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
    # One program per (token, head) row, as the SGLang kernel runs.
    #
    # Each output element is written once, by one contiguous store of the
    # row: a masked store followed by stride-2 stores into the part it
    # skipped left elements unwritten here. The other half of each
    # (real, imag) pair is the same row read one element to the right or to
    # the left, so every access is a contiguous block under a mask that is
    # a single range, and the parity picks between the two afterwards.
    row = tl.program_id(0).to(tl.int64)
    if HEADS == 1:
        token = row
    else:
        token = row // HEADS
    column = tl.arange(0, B)
    inside = column < D
    odd = (column & 1) == 1
    to_next = column < D - 1
    to_prev = (column >= 1) & inside
    source = X + token * SX0 + (row - token * HEADS) * SX1 + column
    x = tl.load(source, inside, other=0.0).to(tl.float32)
    partner = tl.where(
        odd,
        tl.load(source - 1, to_prev, other=0.0),
        tl.load(source + 1, to_next, other=0.0),
    ).to(tl.float32)
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
        out = tl.where(
            odd,
            x * f_prev - partner * f_same,
            x * f_same + partner * f_next,
        )
    else:
        # Even c: real * f_real - imag * f_imag.
        # Odd c: imag * f_real + real * f_imag.
        out = tl.where(
            odd,
            x * f_prev + partner * f_same,
            x * f_same - partner * f_next,
        )
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
        b = 1 << (d - 1).bit_length()
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

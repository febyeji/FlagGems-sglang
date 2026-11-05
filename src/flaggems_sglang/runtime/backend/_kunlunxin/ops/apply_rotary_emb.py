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
    PAIRS: tl.constexpr,
):
    # One program per (token, head) row, as the SGLang kernel runs.
    #
    # The row is one contiguous [pairs, 2] block, read once and written
    # once: no access leaves the row, and the halves of each pair are taken
    # apart and put back together in registers, a product with a 0/1
    # selector summed over the pair. Reading the pair partner one element
    # to the side instead left the last column of every row wrong here,
    # where that read is the one that leaves the row.
    row = tl.program_id(0).to(tl.int64)
    if HEADS == 1:
        token = row
    else:
        token = row // HEADS
    pair = tl.arange(0, PAIRS)
    part = tl.arange(0, 2)
    lane = pair[:, None] * 2 + part[None, :]
    take_real = (part == 0).to(tl.float32)[None, :]
    take_imag = (part == 1).to(tl.float32)[None, :]
    mask = lane < D
    if HAS_P:
        position = tl.load(P + token).to(tl.int64)
    else:
        position = token
    source = X + token * SX0 + (row - token * HEADS) * SX1 + lane
    x = tl.load(source, mask, other=0.0).to(tl.float32)
    x_real = tl.sum(x * take_real, 1)
    x_imag = tl.sum(x * take_imag, 1)
    # The table is interleaved like the row: (real, imag) per pair.
    f = tl.load(F + position * FS + lane, mask, other=0.0)
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
            1 << ((d + 1) // 2 - 1).bit_length(),
            grid=(n,),
            warmup=False,
            num_warps=4,
            isCloseMemoryAsync=False,
        )
    return out


solution = apply_rotary_emb

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
    N: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    FS: tl.constexpr,
    HAS_P: tl.constexpr,
    INVERSE: tl.constexpr,
    PAIRS: tl.constexpr,
    GRID: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    # The trip count is fixed at compile time and each pass takes one row
    # with a scalar index, which is the shape this chip's compiler takes.
    #
    # The row is one contiguous [pairs, 2] block, read once and written once
    # under a mask that is an upper bound. Stride-2 stores into a row left
    # it wrong here, and reads shifted by one element are misaligned, so the
    # halves of each pair are taken apart and put back together in
    # registers: a product with a 0/1 selector summed over the pair is
    # exact.
    pair = tl.arange(0, PAIRS)
    part = tl.arange(0, 2)
    lane = pair[:, None] * 2 + part[None, :]
    take_real = (part == 0).to(tl.float32)[None, :]
    take_imag = (part == 1).to(tl.float32)[None, :]
    for chunk in range(CHUNKS):
        row = (tl.program_id(0) + chunk * GRID).to(tl.int64)
        if HEADS == 1:
            token = row
        else:
            token = row // HEADS
        # The rows divide among the programs when N is a multiple of GRID,
        # and then no access needs a row mask.
        if N % GRID == 0:
            mask = lane < D
            if HAS_P:
                position = tl.load(P + token).to(tl.int64)
            else:
                position = token
        else:
            live = row < N
            mask = (lane < D) & live
            if HAS_P:
                position = tl.load(P + token, live, other=0).to(tl.int64)
            else:
                position = token
        base = token * SX0 + (row - token * HEADS) * SX1
        x = tl.load(X + base + lane, mask, other=0.0).to(tl.float32)
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
        grid = min(n, CORES)
        _rows.run(
            x,
            freqs,
            positions,
            out,
            n,
            heads,
            d,
            x.stride(0),
            x.stride(1) if x.ndim == 3 else 0,
            freqs.stride(0),
            positions is not None,
            inverse,
            1 << ((d + 1) // 2 - 1).bit_length(),
            grid,
            (n + grid - 1) // grid,
            grid=(grid,),
            warmup=False,
        )
    return out


solution = apply_rotary_emb

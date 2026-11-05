# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_norm_rope_inplace"]


@triton.jit
def _row(
    KV,
    W,
    F,
    P,
    OUT,
    D: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    FS: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_P: tl.constexpr,
    B: tl.constexpr,
):
    # One program per row, as the SGLang kernel runs.
    #
    # Each output element is written once, by one contiguous store of the
    # row: a masked store followed by stride-2 stores into the part it
    # skipped left elements unwritten here. The other half of each
    # (real, imag) pair is the same row read one element to the right or
    # to the left, so every access is a contiguous block under a mask that
    # is a single range, and the parity picks between the two afterwards.
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, B)
    inside = column < D
    t = column - (D - R)
    tail = (column >= D - R) & inside
    odd = (t & 1) == 1
    to_next = (column >= D - R) & (column < D - 1)
    to_prev = (column >= D - R + 1) & inside
    source = KV + row * S + column
    x = tl.load(source, inside, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    partner = tl.where(
        odd,
        tl.load(source - 1, to_prev, other=0.0),
        tl.load(source + 1, to_next, other=0.0),
    ).to(tl.float32)
    y = x * inv
    y_pair = partner * inv
    if HAS_W:
        y = y * tl.load(W + column, inside, other=0.0).to(tl.float32)
        y_pair = y_pair * tl.where(
            odd,
            tl.load(W + column - 1, to_prev, other=0.0),
            tl.load(W + column + 1, to_next, other=0.0),
        ).to(tl.float32)
    if HAS_P:
        position = tl.load(P + row).to(tl.int64)
    else:
        position = row
    # The table is interleaved like the row, so element t of the tail reads
    # entry t: the real part on even t, the imaginary part on odd t.
    freq = F + position * FS + t
    f_same = tl.load(freq, tail, other=0.0)
    f_next = tl.load(freq + 1, to_next, other=0.0)
    f_prev = tl.load(freq - 1, to_prev, other=0.0)
    # Even t: real * f_real - imag * f_imag.
    # Odd t: imag * f_real + real * f_imag.
    rotated = tl.where(
        odd,
        y * f_prev + y_pair * f_same,
        y * f_same - y_pair * f_next,
    )
    out = tl.where(tail, rotated, y)
    tl.store(OUT + row * D + column, out.to(OUT.dtype.element_ty), inside)


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
        b = 1 << (d - 1).bit_length()
        _row.run(
            kv,
            weight,
            freqs,
            positions,
            out,
            d,
            freqs_cis.shape[-1] * 2,
            kv.stride(0),
            freqs.stride(0),
            eps,
            weight is not None,
            positions is not None,
            b,
            grid=(m,),
            warmup=False,
            num_warps=4 if b <= 4096 else 8,
            isCloseMemoryAsync=False,
        )
    return out


solution = fused_norm_rope_inplace

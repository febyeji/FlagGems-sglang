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
def _rows(
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
    B: tl.constexpr,
    H: tl.constexpr,
    GRID: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    # The trip count is fixed at compile time and each pass takes one row
    # with a scalar index, which is the shape this chip's compiler takes.
    # Every access is affine: the whole row once, then the rope tail again
    # as two stride-2 halves, the same accesses the SGLang kernel makes.
    column = tl.arange(0, B)
    inside = column < D
    plain = column < D - R
    half = tl.arange(0, H)
    paired = half < R // 2
    tail = D - R + 2 * half
    if HAS_W:
        w = tl.load(W + column, inside, other=0.0).to(tl.float32)
        w_real = tl.load(W + tail, paired, other=0.0).to(tl.float32)
        w_imag = tl.load(W + tail + 1, paired, other=0.0).to(tl.float32)
    for chunk in range(CHUNKS):
        row = (tl.program_id(0) + chunk * GRID).to(tl.int64)
        # The rows divide among the programs when M is a multiple of GRID,
        # and then no access needs a row mask.
        if M % GRID == 0:
            row_mask = inside
            plain_mask = plain
            pair_mask = paired
            if HAS_P:
                position = tl.load(P + row).to(tl.int64)
            else:
                position = row
        else:
            live = row < M
            row_mask = inside & live
            plain_mask = plain & live
            pair_mask = paired & live
            if HAS_P:
                position = tl.load(P + row, live, other=0).to(tl.int64)
            else:
                position = row
        x = tl.load(KV + row * S + column, row_mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        y = x * inv
        if HAS_W:
            y = y * w
        tl.store(
            OUT + row * D + column, y.to(OUT.dtype.element_ty), plain_mask
        )
        x_real = tl.load(KV + row * S + tail, pair_mask, other=0.0).to(
            tl.float32
        )
        x_imag = tl.load(KV + row * S + tail + 1, pair_mask, other=0.0).to(
            tl.float32
        )
        x_real = x_real * inv
        x_imag = x_imag * inv
        if HAS_W:
            x_real = x_real * w_real
            x_imag = x_imag * w_imag
        f_real = tl.load(F + position * FS + 2 * half, pair_mask, other=0.0)
        f_imag = tl.load(
            F + position * FS + 2 * half + 1, pair_mask, other=0.0
        )
        tl.store(
            OUT + row * D + tail,
            (x_real * f_real - x_imag * f_imag).to(OUT.dtype.element_ty),
            pair_mask,
        )
        tl.store(
            OUT + row * D + tail + 1,
            (x_real * f_imag + x_imag * f_real).to(OUT.dtype.element_ty),
            pair_mask,
        )


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    """RMSNorm over the row, then RoPE on its interleaved rope tail.

    The SGLang kernel rewrites ``kv`` in place; this writes a new tensor and
    leaves ``kv`` as it was, which is what the task's reference returns.
    """
    m, d = kv.shape
    freqs = (
        freqs_cis.view(torch.float32)
        if freqs_cis.stride(-1) == 1
        else torch.view_as_real(freqs_cis)
    )
    out = torch.empty_like(kv)
    if m:
        r = freqs_cis.shape[-1] * 2
        grid = min(m, CORES)
        _rows.run(
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
            1 << (d - 1).bit_length(),
            1 << max(r // 2 - 1, 0).bit_length(),
            grid,
            (m + grid - 1) // grid,
            grid=(grid,),
            warmup=False,
        )
    return out


solution = fused_norm_rope_inplace

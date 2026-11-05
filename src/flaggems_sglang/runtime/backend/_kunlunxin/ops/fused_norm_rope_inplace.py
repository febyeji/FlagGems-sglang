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
    H: tl.constexpr,
):
    # One program per row, as the SGLang kernel runs, with the row read once
    # as a contiguous block and the rope tail again as two stride-2 halves.
    # The compiler here is told which accesses are contiguous, and loads are
    # clamped instead of masked where the row is not a power of two wide.
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, B)
    if B == D:
        x = tl.load(
            KV + row * S + column, offset_state_policy="continuous"
        ).to(tl.float32)
    else:
        x = tl.load(KV + row * S + tl.minimum(column, D - 1)).to(tl.float32)
        x = tl.where(column < D, x, 0.0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = x * inv
    if HAS_W:
        if B == D:
            w = tl.load(W + column, offset_state_policy="continuous")
        else:
            w = tl.load(W + tl.minimum(column, D - 1))
        y = y * w.to(tl.float32)
    tl.store(
        OUT + row * D + column, y.to(OUT.dtype.element_ty), column < D - R
    )
    half = tl.arange(0, H)
    if H == R // 2:
        pair = half
    else:
        pair = tl.minimum(half, R // 2 - 1)
    tail = D - R + 2 * pair
    x_real = tl.load(KV + row * S + tail).to(tl.float32) * inv
    x_imag = tl.load(KV + row * S + tail + 1).to(tl.float32) * inv
    if HAS_W:
        x_real = x_real * tl.load(W + tail).to(tl.float32)
        x_imag = x_imag * tl.load(W + tail + 1).to(tl.float32)
    if HAS_P:
        position = tl.load(P + row).to(tl.int64)
    else:
        position = row
    f_real = tl.load(F + position * FS + 2 * pair)
    f_imag = tl.load(F + position * FS + 2 * pair + 1)
    real = (x_real * f_real - x_imag * f_imag).to(OUT.dtype.element_ty)
    imag = (x_real * f_imag + x_imag * f_real).to(OUT.dtype.element_ty)
    if H == R // 2:
        tl.store(OUT + row * D + tail, real)
        tl.store(OUT + row * D + tail + 1, imag)
    else:
        tl.store(OUT + row * D + tail, real, half < R // 2)
        tl.store(OUT + row * D + tail + 1, imag, half < R // 2)


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
        b = 1 << (d - 1).bit_length()
        _row.run(
            kv,
            weight,
            freqs,
            positions,
            out,
            d,
            r,
            kv.stride(0),
            freqs.stride(0),
            eps,
            weight is not None,
            positions is not None,
            b,
            1 << max(r // 2 - 1, 0).bit_length(),
            grid=(m,),
            warmup=False,
            num_warps=4 if b <= 4096 else 8,
            isCloseMemoryAsync=False,
        )
    return out


solution = fused_norm_rope_inplace

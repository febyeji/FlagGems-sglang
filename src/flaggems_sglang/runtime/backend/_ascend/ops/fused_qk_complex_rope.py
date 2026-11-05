# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_qk_complex_rope"]

# One program per vector core, each walking the (token, head) rows in a
# loop whose trip count is a compile-time constant. The whole row is one
# flat tile with a scalar row index, which is the shape this chip's
# compiler takes: the generic file's two-dimensional head block is a
# wider tile than a vector core wants.
PROGRAMS = 40
NUM_WARPS = 1


@triton.jit
def _fused_qk_complex_rope_kernel(
    Q,
    K,
    F,
    QO,
    KO,
    ROWS: tl.constexpr,
    ITERS: tl.constexpr,
    STEP: tl.constexpr,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAIRS: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    FT: tl.constexpr,
    EXACT: tl.constexpr,
    WHOLE: tl.constexpr,
):
    pair = tl.arange(0, PAIRS)
    if EXACT:
        mask = None
    else:
        mask = pair < HALF
    dtype = QO.dtype.element_ty
    for step in range(ITERS):
        row = tl.program_id(0) + step * STEP
        # The grid divides the rows when WHOLE, and then the loop needs
        # no bound of its own.
        if WHOLE or row < ROWS:
            token = row // HEADS
            head = row - token * HEADS
            freq = F + token * FT + pair * 2
            cos = tl.load(freq, mask=mask)
            sin = tl.load(freq + 1, mask=mask)
            q_in = Q + token * QT + head * QH + pair * (2 * QD)
            k_in = K + token * KT + head * KH + pair * (2 * KD)
            q_real = tl.load(q_in, mask=mask).to(tl.float32)
            q_imag = tl.load(q_in + QD, mask=mask).to(tl.float32)
            k_real = tl.load(k_in, mask=mask).to(tl.float32)
            k_imag = tl.load(k_in + KD, mask=mask).to(tl.float32)
            out = row * (2 * HALF) + pair * 2
            tl.store(
                QO + out, (q_real * cos - q_imag * sin).to(dtype), mask=mask
            )
            tl.store(
                QO + out + 1,
                (q_real * sin + q_imag * cos).to(dtype),
                mask=mask,
            )
            tl.store(
                KO + out, (k_real * cos - k_imag * sin).to(dtype), mask=mask
            )
            tl.store(
                KO + out + 1,
                (k_real * sin + k_imag * cos).to(dtype),
                mask=mask,
            )


def fused_qk_complex_rope(q, k, freqs_cis):
    """Rotate Q and K by one complex frequency row per token.

    Each head row is `D/2` complex numbers built from adjacent elements,
    multiplied by the token's frequency row and written back interleaved.
    Q and K go through one launch, and the outputs are contiguous even
    when the inputs are strided views of a packed QKV projection.
    """
    shape = q.shape
    heads = shape[-2]
    half = shape[-1] // 2
    q_flat = q if q.ndim == 3 else q.view(-1, heads, half * 2)
    k_flat = k if k.ndim == 3 else k.view(-1, heads, half * 2)
    freqs = torch.view_as_real(freqs_cis).view(-1, half, 2)
    tokens = q_flat.shape[0]
    q_out = torch.empty(q_flat.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_flat.shape, dtype=k.dtype, device=k.device)
    if tokens:
        rows = tokens * heads
        programs = min(rows, PROGRAMS)
        _fused_qk_complex_rope_kernel[(programs,)](
            q_flat,
            k_flat,
            freqs,
            q_out,
            k_out,
            rows,
            triton.cdiv(rows, programs),
            programs,
            heads,
            half,
            triton.next_power_of_2(half),
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            freqs.stride(0),
            triton.next_power_of_2(half) == half,
            rows % programs == 0,
            num_warps=NUM_WARPS,
        )
    return q_out.view(shape), k_out.view(shape)


solution = fused_qk_complex_rope

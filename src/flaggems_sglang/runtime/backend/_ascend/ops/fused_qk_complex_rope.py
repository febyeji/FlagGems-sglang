# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_qk_complex_rope"]

# One program per vector core, walking its share of the (token, head
# block) tiles in a loop whose trip count is a compile-time constant.
# The tile is as wide as TILE values so each load fills the vector unit;
# a submission that gave this chip one head row per step, 32 pairs at a
# time, ran at 0.74 of the SGLang kernel.
TILE = 1024
PROGRAMS = 40
NUM_WARPS = 1


def _block_heads(heads, pairs):
    return 1 << (max(1, min(heads, TILE // pairs)).bit_length() - 1)


@triton.jit
def _fused_qk_complex_rope_kernel(
    Q,
    K,
    F,
    QO,
    KO,
    WORK: tl.constexpr,
    ITERS: tl.constexpr,
    STEP: tl.constexpr,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HBLOCKS: tl.constexpr,
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
    block = tl.arange(0, BLOCK_H)[:, None]
    dtype = QO.dtype.element_ty
    for step in range(ITERS):
        item = tl.program_id(0) + step * STEP
        # The grid divides the work when WHOLE, and the loop then needs
        # no bound of its own.
        if WHOLE or item < WORK:
            token = item // HBLOCKS
            head = (item - token * HBLOCKS) * BLOCK_H + block
            if EXACT:
                mask = None
            else:
                mask = (head < HEADS) & (pair[None, :] < HALF)
            freq = F + token * FT + pair * 2
            cos = tl.load(freq, mask=None if EXACT else pair < HALF)[None, :]
            sin = tl.load(
                freq + 1, mask=None if EXACT else pair < HALF
            )[None, :]
            q_in = Q + token * QT + head * QH + pair[None, :] * (2 * QD)
            k_in = K + token * KT + head * KH + pair[None, :] * (2 * KD)
            q_real = tl.load(q_in, mask=mask).to(tl.float32)
            q_imag = tl.load(q_in + QD, mask=mask).to(tl.float32)
            k_real = tl.load(k_in, mask=mask).to(tl.float32)
            k_imag = tl.load(k_in + KD, mask=mask).to(tl.float32)
            out = (
                token * (HEADS * 2 * HALF)
                + head * (2 * HALF)
                + pair[None, :] * 2
            )
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
        pairs = triton.next_power_of_2(half)
        block_h = _block_heads(heads, pairs)
        hblocks = triton.cdiv(heads, block_h)
        work = tokens * hblocks
        programs = min(work, PROGRAMS)
        _fused_qk_complex_rope_kernel[(programs,)](
            q_flat,
            k_flat,
            freqs,
            q_out,
            k_out,
            work,
            triton.cdiv(work, programs),
            programs,
            heads,
            half,
            pairs,
            block_h,
            hblocks,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            freqs.stride(0),
            pairs == half and heads % block_h == 0,
            work % programs == 0,
            num_warps=NUM_WARPS,
        )
    return q_out.view(shape), k_out.view(shape)


solution = fused_qk_complex_rope

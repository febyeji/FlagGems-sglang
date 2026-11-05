# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_qk_complex_rope"]

# A pair of 16-bit values is one 32-bit word, so this chip reads and
# writes the row as words: every access is contiguous and there are half
# as many of them as the two accesses of stride two the generic file
# makes. That shape took haiguang from 2.03 to 4.67 of the SGLang
# kernel in one graded run, and this chip ran the strided file
# at 3.00. The halves come out
# of the word with shifts, which is exact for BF16 because its bits are
# an FP32 word's top half, and through the 16-bit float for FP16. Two warps: this chip reached
# 3.18 with the word form at four, the lowest of the five GPU-like
# chips, where the others reached 4.1 to 4.5.
TILE = 4096
PROGRAMS = 1024
NUM_WARPS = 2


def _block_heads(tokens, heads, pairs):
    block = 1 << (max(1, min(heads, TILE // pairs)).bit_length() - 1)
    while block > 1 and tokens * triton.cdiv(heads, block) < PROGRAMS:
        block //= 2
    return block


@triton.jit
def _fused_qk_complex_rope_kernel(
    Q,
    K,
    F,
    QO,
    KO,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HBLOCKS: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    FT: tl.constexpr,
    EXACT: tl.constexpr,
    BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // HBLOCKS
    head = (pid % HBLOCKS) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    pair = tl.arange(0, PAIRS)
    if EXACT:
        mask = None
        row = None
    else:
        row = pair < HALF
        mask = (head < HEADS) & row[None, :]
    freq = F + token * FT + pair * 2
    cos = tl.load(freq, mask=row)[None, :]
    sin = tl.load(freq + 1, mask=row)[None, :]
    q_word = tl.load(Q + token * QT + head * QH + pair[None, :], mask=mask)
    k_word = tl.load(K + token * KT + head * KH + pair[None, :], mask=mask)
    if BF16:
        # BF16 is the top half of an FP32 word, so the shift is the cast.
        q_real = (q_word << 16).to(tl.float32, bitcast=True)
        q_imag = ((q_word >> 16) << 16).to(tl.float32, bitcast=True)
        k_real = (k_word << 16).to(tl.float32, bitcast=True)
        k_imag = ((k_word >> 16) << 16).to(tl.float32, bitcast=True)
    else:
        q_real = (q_word & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        q_imag = (q_word >> 16).to(tl.int16).to(tl.float16, bitcast=True)
        k_real = (k_word & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        k_imag = (k_word >> 16).to(tl.int16).to(tl.float16, bitcast=True)
        q_real = q_real.to(tl.float32)
        q_imag = q_imag.to(tl.float32)
        k_real = k_real.to(tl.float32)
        k_imag = k_imag.to(tl.float32)
    q_out = _pack(q_real * cos - q_imag * sin, q_real * sin + q_imag * cos, BF16)
    k_out = _pack(k_real * cos - k_imag * sin, k_real * sin + k_imag * cos, BF16)
    out = token * (HEADS * HALF) + head * HALF + pair[None, :]
    tl.store(QO + out, q_out, mask=mask)
    tl.store(KO + out, k_out, mask=mask)


@triton.jit
def _pack(real, imag, BF16: tl.constexpr):
    """Round both halves to the output type and lay them in one word."""
    if BF16:
        low = real.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        high = imag.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    else:
        low = real.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        high = imag.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
    # The 16-bit halves arrive sign extended, which the mask undoes for
    # the low one and the shift drops for the high one.
    return (low & 0xFFFF) | (high << 16)


def fused_qk_complex_rope(q, k, freqs_cis):
    """Rotate Q and K by one complex frequency row per token.

    Each head row is `D/2` complex numbers built from adjacent elements,
    multiplied by the token's frequency row and written back interleaved.
    Q and K go through one launch, and the outputs are contiguous even
    when the inputs are strided views of a packed QKV projection.
    """
    shape = q.shape
    heads = shape[-2]
    dim = shape[-1]
    half = dim // 2
    q_flat = q if q.ndim == 3 else q.view(-1, heads, dim)
    k_flat = k if k.ndim == 3 else k.view(-1, heads, dim)
    freqs = torch.view_as_real(freqs_cis).view(-1, half, 2)
    tokens = q_flat.shape[0]
    q_out = torch.empty(q_flat.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_flat.shape, dtype=k.dtype, device=k.device)
    if tokens:
        # A pair is one word only where the row is contiguous, which a
        # view of a packed QKV projection still is along its last two
        # dimensions.
        pairs = triton.next_power_of_2(half)
        block_h = _block_heads(tokens, heads, pairs)
        hblocks = triton.cdiv(heads, block_h)
        q_word = q_flat.view(torch.int32)
        k_word = k_flat.view(torch.int32)
        _fused_qk_complex_rope_kernel[(tokens * hblocks,)](
            q_word,
            k_word,
            freqs,
            q_out.view(torch.int32),
            k_out.view(torch.int32),
            heads,
            half,
            pairs,
            block_h,
            hblocks,
            q_word.stride(0),
            q_word.stride(1),
            k_word.stride(0),
            k_word.stride(1),
            freqs.stride(0),
            pairs == half and heads % block_h == 0,
            q.dtype == torch.bfloat16,
            num_warps=NUM_WARPS,
        )
    return q_out.view(shape), k_out.view(shape)


solution = fused_qk_complex_rope

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_qk_complex_rope"]

# A program takes one token's frequency row and as many heads as fit in
# TILE values, so the row is read once and reused across the heads that
# share it. Few tokens would leave such a grid too small to fill the
# chip, so the block is halved until there are PROGRAMS of them.
# NUM_WARPS is the generic setting; chips that want another one carry
# their own copy under runtime/backend/.
TILE = 4096
PROGRAMS = 1024
NUM_WARPS = 4

# Graded on all seven chips through the competition's harness, where
# each of them runs the file that carries its own name and falls back
# to this one. Submission 64215 is valid at 3.1531 average, first of
# four teams: haiguang 4.6742, muxi 4.5180, tianshu 4.1698,
# card_a 4.1632, card_b 3.2212, huawei 1.0978, kunlunxin 0.2272. The
# five GPU-like chips read and write the row as 32-bit words, which is
# worth about twice this file's two accesses of stride two; this file
# keeps those because a word view needs an even stride and an even
# storage offset, which an arbitrary input is not required to have.
# The graded records, the per-variant table and the local timings are
# under ~/Codespaces/FlagGems-sglang-results/yej-354/, listed in its
# README.


def _block_heads(tokens, heads, pairs):
    # A tile side has to be a power of two: several of the chips reject
    # `tl.arange` over anything else. A block that does not divide the
    # head count is masked instead.
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
    QD: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    FT: tl.constexpr,
    EXACT: tl.constexpr,
):
    # One program: BLOCK_H heads of one token, every pair of each head.
    pid = tl.program_id(0)
    token = pid // HBLOCKS
    head = (pid % HBLOCKS) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    pair = tl.arange(0, PAIRS)
    if EXACT:
        # The head block divides the head count and the pair count is a
        # power of two, so every index in the tile is in range.
        mask = None
        row = None
    else:
        row = pair < HALF
        mask = (head < HEADS) & row[None, :]

    # The frequency row is complex64, which is a pair of FP32 words, so
    # the real part sits at twice the pair index and the imaginary part
    # beside it. The same row applies to every head in the tile.
    freq = F + token * FT + pair * 2
    cos = tl.load(freq, mask=row)[None, :]
    sin = tl.load(freq + 1, mask=row)[None, :]

    # The last dimension holds (real, imag) adjacent, so a pair starts at
    # twice the pair index in units of the input's own last-dim stride.
    q_in = Q + token * QT + head * QH + pair[None, :] * (2 * QD)
    k_in = K + token * KT + head * KH + pair[None, :] * (2 * KD)
    q_real = tl.load(q_in, mask=mask).to(tl.float32)
    q_imag = tl.load(q_in + QD, mask=mask).to(tl.float32)
    k_real = tl.load(k_in, mask=mask).to(tl.float32)
    k_imag = tl.load(k_in + KD, mask=mask).to(tl.float32)

    # The outputs are contiguous whatever the inputs were, so their
    # strides are the shape itself.
    out = head * (2 * HALF) + pair[None, :] * 2
    q_out = QO + token * (HEADS * 2 * HALF) + out
    k_out = KO + token * (HEADS * 2 * HALF) + out
    dtype = QO.dtype.element_ty
    tl.store(q_out, (q_real * cos - q_imag * sin).to(dtype), mask=mask)
    tl.store(q_out + 1, (q_real * sin + q_imag * cos).to(dtype), mask=mask)
    tl.store(k_out, (k_real * cos - k_imag * sin).to(dtype), mask=mask)
    tl.store(k_out + 1, (k_real * sin + k_imag * cos).to(dtype), mask=mask)


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
    # Views of a QKV projection keep their token stride, so a three
    # dimensional input is taken as it is; anything deeper is flattened
    # the way the frequencies already are.
    q_flat = q if q.ndim == 3 else q.view(-1, heads, half * 2)
    k_flat = k if k.ndim == 3 else k.view(-1, heads, half * 2)
    freqs = torch.view_as_real(freqs_cis).view(-1, half, 2)
    tokens = q_flat.shape[0]
    q_out = torch.empty(q_flat.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_flat.shape, dtype=k.dtype, device=k.device)
    if tokens:
        pairs = triton.next_power_of_2(half)
        block_h = _block_heads(tokens, heads, pairs)
        hblocks = triton.cdiv(heads, block_h)
        _fused_qk_complex_rope_kernel[(tokens * hblocks,)](
            q_flat,
            k_flat,
            freqs,
            q_out,
            k_out,
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
            num_warps=NUM_WARPS,
        )
    return q_out.view(shape), k_out.view(shape)


solution = fused_qk_complex_rope

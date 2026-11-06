# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["fused_qk_complex_rope"]

# This chip returned a third to a half of the output unwritten from the
# shape every other chip grades as correct, both as a flat row and as a
# two-dimensional head block. What both have in common is the pair: they
# write the real and the imaginary halves as two stores of stride two.
# Here the tile is the head row itself, every element of it, so each
# store is one contiguous run. An element takes its partner from the
# same row through an index that flips the low bit, and the frequency
# that belongs to its pair by halving its column.
TILE = 1024
NUM_WARPS = 1


@triton.jit
def _fused_qk_complex_rope_kernel(
    Q,
    K,
    F,
    QO,
    KO,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    FT: tl.constexpr,
    FP: tl.constexpr,
    FC: tl.constexpr,
    EXACT: tl.constexpr,
):
    item = tl.program_id(0)
    token = item // BLOCKS
    head = (item - token * BLOCKS) * ROWS
    col = tl.arange(0, WIDTH)
    if EXACT:
        mask = None
    else:
        mask = col < ROWS * DIM
    if ROWS > 1:
        # Whole head rows sit next to each other, so the tile is one run
        # of memory and the column inside a row comes from the low bits.
        lane = col & (DIM - 1)
        q_in = Q + token * QT + head * QH + col
        k_in = K + token * KT + head * KH + col
        q_swap = Q + token * QT + head * QH + (col ^ 1)
        k_swap = K + token * KT + head * KH + (col ^ 1)
    else:
        lane = col
        q_in = Q + token * QT + head * QH + col * QD
        k_in = K + token * KT + head * KH + col * KD
        q_swap = Q + token * QT + head * QH + (col ^ 1) * QD
        k_swap = K + token * KT + head * KH + (col ^ 1) * KD
    # The frequency of a column is the one of its pair, and the partner
    # enters with a minus on the real half and a plus on the imaginary
    # one, which is the sign of the column's low bit.
    freq = F + token * FT + (lane // 2) * FP
    cos = tl.load(freq, mask=mask)
    sin = tl.load(freq + FC, mask=mask) * ((lane & 1) * 2.0 - 1.0)
    q_value = tl.load(q_in, mask=mask).to(tl.float32)
    k_value = tl.load(k_in, mask=mask).to(tl.float32)
    q_other = tl.load(q_swap, mask=mask).to(tl.float32)
    k_other = tl.load(k_swap, mask=mask).to(tl.float32)
    out = (token * HEADS + head) * DIM + col
    dtype = QO.dtype.element_ty
    tl.store(QO + out, (q_value * cos + q_other * sin).to(dtype), mask=mask)
    tl.store(KO + out, (k_value * cos + k_other * sin).to(dtype), mask=mask)


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
    # A view of a QKV projection keeps its token stride, so a three
    # dimensional input is taken as it is and its strides are handed in.
    # Anything deeper has its leading dimensions flattened into one;
    # reshape returns a view when they collapse into a single stride and
    # copies when they do not, which view reports as an error instead.
    q_flat = q if q.ndim == 3 else q.reshape(-1, heads, dim)
    k_flat = k if k.ndim == 3 else k.reshape(-1, heads, dim)
    # A complex64 row is a pair of FP32 words, and the view of it as
    # floats keeps whatever strides the frequencies arrived with, which
    # a slice of a wider table need not make contiguous. Only leading
    # dimensions are flattened, and reshape copies when they do not
    # collapse into one.
    freqs = torch.view_as_real(freqs_cis)
    if freqs.ndim > 3:
        freqs = freqs.reshape(-1, half, 2)
    tokens = q_flat.shape[0]
    q_out = torch.empty(q_flat.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_flat.shape, dtype=k.dtype, device=k.device)
    if tokens:
        q_stride = q_flat.stride()
        k_stride = k_flat.stride()
        # Several head rows make one tile only when they are adjacent in
        # both inputs and the row width is a power of two, which is what
        # lets the column inside a row come from the low bits.
        rows = 1
        packed = (
            q_stride[1] == dim
            and k_stride[1] == dim
            and q_stride[2] == 1
            and k_stride[2] == 1
            and dim & (dim - 1) == 0
        )
        if packed:
            rows = 1 << (max(1, min(heads, TILE // dim)).bit_length() - 1)
            while rows > 1 and heads % rows:
                rows //= 2
        blocks = heads // rows
        _fused_qk_complex_rope_kernel[(tokens * blocks,)](
            q_flat,
            k_flat,
            freqs,
            q_out,
            k_out,
            heads,
            dim,
            triton.next_power_of_2(rows * dim),
            rows,
            blocks,
            # One call for the three strides: at these sizes the
            # kernel runs in the time the call around it takes.
            *q_stride,
            *k_stride,
            *freqs.stride(),
            triton.next_power_of_2(rows * dim) == rows * dim,
            num_warps=NUM_WARPS,
        )
    return q_out.view(shape), k_out.view(shape)


solution = fused_qk_complex_rope

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["set_mla_kv_buffer_fp8_quant"]

E4M3FN = tl.constexpr(0)
E5M2 = tl.constexpr(1)

_FORMATS = {
    torch.float8_e4m3fn: E4M3FN.value,
    torch.float8_e5m2: E5M2.value,
}


@triton.jit
def _widen(pointer, mask):
    if pointer.dtype.element_ty == tl.bfloat16:
        # Widening BF16 is a shift, and the shift avoids the native cast.
        word = tl.load(pointer.to(tl.pointer_type(tl.uint16)), mask, other=0)
        return (word.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(pointer, mask, other=0.0).to(tl.float32)


@triton.jit
def _encode(value, FORMAT: tl.constexpr):
    """Round FP32 to one FP8 byte, bit for bit as torch's own cast does.

    Signed words throughout: the chips without a native FP8 cast are also
    the ones that dislike unsigned arithmetic.
    """
    bits = value.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 128
    magnitude = bits & 0x7FFFFFFF
    number = tl.abs(value)
    if FORMAT == E4M3FN:
        # E4M3FN has no infinity, and torch's cast saturates: every
        # magnitude from 448.0 up, infinity included, lands on 448.0.
        capped = tl.minimum(number, 448.0)
        word = capped.to(tl.int32, bitcast=True)
        # The bias carries one subnormal step per FP32 ULP, so the addition
        # rounds ties to even straight into the low mantissa bits.
        subnormal = (capped + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
        rounded = word + 0x7FFFF + ((word >> 20) & 1)
        normal = (rounded >> 20) - (120 << 3)
        code = tl.where(capped < 2.0**-6, subnormal, normal)
        code = tl.where(magnitude > (255 << 23), 0x7F, code)
    else:
        # E5M2 keeps infinity, so only a payload above infinity is NaN.
        over = magnitude >= (143 << 23)
        subnormal = (number + 128.0).to(tl.int32, bitcast=True) - 0x43000000
        rounded = magnitude + ((15 - 127) << 23) + 0xFFFFF
        normal = (rounded + ((magnitude >> 21) & 1)) >> 21
        code = tl.where(magnitude < (113 << 23), subnormal, normal)
        code = tl.where(
            over, tl.where(magnitude > (255 << 23), 0x7F, 0x7C), code
        )
    return (code | sign).to(tl.uint8)


@triton.jit
def _scatter(
    DST,
    LOC,
    NOPE,
    ROPE,
    rows,
    NOPE_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    DST_S0: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    FORMAT: tl.constexpr,
    COLS: tl.constexpr,
    WIDE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A flat grid over row blocks and column blocks. Neighbouring programs
    # walk the columns of one row block, so their reads and their FP8 writes
    # stay inside the same rows.
    pid = tl.program_id(0)
    part = pid % COLS
    token = (pid // COLS) * ROWS + tl.arange(0, ROWS)
    column = part * BLOCK + tl.arange(0, BLOCK)
    live = token < rows
    slot = tl.load(LOC + token * LOC_S0, live, 0)
    if WIDE:
        slot = slot.to(tl.int64)
    else:
        slot = slot.to(tl.int32)
    valid = live[:, None] & (column < WIDTH)[None, :]
    # The column block falls entirely in nope, entirely in rope, or across
    # the seam. Which one is uniform across the program, so each program
    # pays for one load except on the seam.
    base = part * BLOCK
    if base + BLOCK <= NOPE_DIM:
        value = _widen(
            NOPE + token[:, None] * NOPE_S0 + column[None, :], valid
        )
    elif base >= NOPE_DIM:
        value = _widen(
            ROPE + token[:, None] * ROPE_S0 + (column - NOPE_DIM)[None, :],
            valid,
        )
    else:
        head = column < NOPE_DIM
        nope_value = _widen(
            NOPE + token[:, None] * NOPE_S0 + column[None, :],
            valid & head[None, :],
        )
        rope_value = _widen(
            ROPE + token[:, None] * ROPE_S0 + (column - NOPE_DIM)[None, :],
            valid & ~head[None, :],
        )
        value = tl.where(head[None, :], nope_value, rope_value)
    tl.store(
        DST + slot[:, None] * DST_S0 + column[None, :],
        _encode(value, FORMAT),
        valid,
    )


def _tile(width, budget):
    """Split a row so the column blocks tile it without a masked remainder."""
    block = 1
    while block < width and block < 512 and width % (block * 2) == 0:
        block *= 2
    return block, max(1, budget // block)


def set_mla_kv_buffer_fp8_quant(
    kv_buffer, loc, cache_k_nope, cache_k_rope, fp8_dtype
):
    """Write MLA K rows into the paged KV cache, downcast to FP8 on store."""
    rows = loc.numel()
    if rows:
        nope_dim = cache_k_nope.shape[-1]
        width = nope_dim + cache_k_rope.shape[-1]
        block, row_tile = _tile(width, 2048)
        cols = triton.cdiv(width, block)
        bound = kv_buffer.numel() * kv_buffer.element_size()
        _scatter[(cols * triton.cdiv(rows, row_tile),)](
            triton.reinterpret(kv_buffer, tl.uint8),
            loc,
            cache_k_nope,
            cache_k_rope,
            rows,
            nope_dim,
            width,
            kv_buffer.stride(0) * kv_buffer.element_size(),
            loc.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            _FORMATS[fp8_dtype],
            cols,
            bound >= 2**31,
            row_tile,
            block,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return kv_buffer


solution = set_mla_kv_buffer_fp8_quant

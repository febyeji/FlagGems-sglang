# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["per_token_group_quant_mla_masked_fp8"]

# 256 is the largest block_m the DeepGEMM masked kernel reads.
ALIGN = 256
# Elements per program, measured on the chip against the SGLang baseline:
#
#    128 (1 group),   1 warp,  separable  0.073x
#   1024 (8 rows),    1 warp,  separable  0.581x
#   1024 (8 rows),    1 warp,  separable  0.582x  grid-stride (59513)
#   1024 (8 groups),  1 warp,  2-D index  invalid_correctness (59491)
#   2048 (16 rows),   1 warp,  separable  invalid_correctness (59520)
#   4096 (32 rows),   1 warp,  separable  invalid_correctness (59501)
#   4096 (32 rows),   4 warps, separable  invalid_correctness (59483)
#
# Eight times the work per program bought eight times the speedup, but
# the grid-stride loop, which cuts programs eightfold while running the
# same number of tile iterations, changed nothing. So the chip charges
# per tile iteration, not per program, and only a wider tile cuts them.
# It refuses every width above 1024, with one warp and with four, so
# 1024 with a separable index is where this kernel stops here.
TILE = 1024


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN with signed words and no native FP8 cast."""
    bits = value.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.minimum(tl.abs(value), 448.0)
    u = magnitude.to(tl.int32, bitcast=True)
    rounded = u + 0x7FFFF + ((u >> 20) & 1)
    normal = (rounded >> 20) - (120 << 3)
    # The bias has one FP8 subnormal step per FP32 ULP, so addition rounds
    # ties to even directly into the low mantissa bits.
    subnormal = (magnitude + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    encoded = tl.where(magnitude < 2.0**-6, subnormal, normal)
    return tl.where(nan, 127, sign | encoded).to(tl.uint8)


@triton.jit
def _quantize(
    X,
    Q,
    S,
    MM,
    EPS,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    TILES: tl.constexpr,
    BLOCKS: tl.constexpr,
    XB: tl.constexpr,
    XR: tl.constexpr,
    XC: tl.constexpr,
    QB: tl.constexpr,
    QR: tl.constexpr,
    SB: tl.constexpr,
    SG: tl.constexpr,
    WIDE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    batch = pid // (TILES * BLOCKS)
    rest = pid % (TILES * BLOCKS)
    # Neighbouring programs walk the K groups of one row block, which keeps
    # their reads and their FP8 writes inside the same rows.
    tile = rest % TILES
    block = rest // TILES
    lane = tl.arange(0, ROWS)
    row = block * ROWS + lane
    span = tl.arange(0, BLOCK)
    column = tile * GROUP + span
    live = row < M
    # Every batch holds the same row count, so the batch's first program
    # writes it here rather than paying for a second launch.
    tl.store(
        MM + batch + lane,
        tl.full([ROWS], M, tl.int32),
        (rest == 0) & (lane == 0),
    )
    # BLOCK rounds GROUP up to a power of two, so the lanes past the end
    # of the group belong to the next one and stay out of both the
    # reduction and the store.
    inside = (span < GROUP) & (column < K)
    valid = live[:, None] & inside[None, :]
    value = tl.load(
        X + batch * XB + row[:, None] * XR + column[None, :] * XC, valid, 0.0
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(value), axis=1), EPS)
    scale = absmax / 448.0
    code = _encode_fp8(tl.div_rn(value, scale[:, None]))
    tl.store(Q + batch * QB + row[:, None] * QR + column[None, :], code, valid)
    # The scales of neighbouring rows sit next to each other in the
    # transposed buffer, so this narrow write stays coalesced.
    tl.store(S + batch * SB + tile * SG + row, scale, live)


def per_token_group_quant_mla_masked_fp8(x, group_size, eps):
    """Quantize an MLA activation per token group into DeepGEMM's layout."""
    if x.dim() != 3:
        raise ValueError("x must be a 3-D MLA activation")
    if group_size <= 0 or x.shape[-1] % group_size:
        raise ValueError("The last dimension must be divisible by group_size")
    batches, rows, width = x.shape
    aligned = (rows + ALIGN - 1) // ALIGN * ALIGN
    tiles = width // group_size
    x_q = x.new_empty((batches, aligned, width), dtype=torch.float8_e4m3fn)
    # Scales are built transposed, the way the masked GEMM reads them.
    x_s = x.new_empty((batches, tiles, aligned), dtype=torch.float32)
    masked_m = torch.empty((batches,), dtype=torch.int32, device=x.device)
    if batches and rows and tiles:
        block = triton.next_power_of_2(group_size)
        row_tile = max(1, TILE // block)
        blocks = triton.cdiv(rows, row_tile)
        programs = batches * tiles * blocks
        bound = max(
            x.numel(),
            batches * aligned * width,
            programs * row_tile * block,
        )
        _quantize[(programs,)](
            x,
            x_q.view(torch.uint8),
            x_s,
            masked_m,
            eps,
            rows,
            width,
            group_size,
            tiles,
            blocks,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            aligned * width,
            width,
            tiles * aligned,
            aligned,
            bound >= 2**31,
            row_tile,
            block,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        # Nothing to quantize, so nothing writes the row count.
        masked_m.fill_(rows)
    return x_q, x_s.transpose(1, 2), masked_m, rows, aligned


solution = per_token_group_quant_mla_masked_fp8

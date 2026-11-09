# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]


@triton.jit
def _encode_fp8(value):
    """Round to E4M3FN, matching torch's saturating round-to-nearest cast."""
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.minimum(tl.abs(value), 448.0)
    u = magnitude.to(tl.uint32, bitcast=True)
    rounded = u + 0x7FFFF + ((u >> 20) & 1)
    normal = (rounded >> 20) - (120 << 3)
    # The bias has one FP8 subnormal step per FP32 ULP, so addition rounds
    # ties to even directly into the low mantissa bits.
    subnormal = (magnitude + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    encoded = tl.where(magnitude < 2.0**-6, subnormal, normal)
    return tl.where(nan, 127, sign | encoded).to(tl.uint8)


@triton.jit
def _silu_quant(
    X,
    M,
    S,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    TOTAL: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program covers ROWS rows of the flattened (expert, token) axis and
    # one column block of the half row. The input rows of a gate column and
    # its up column sit H apart in the contiguous [E, T, 2H] layout.
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    in_range = rows < TOTAL
    rows64 = rows.to(tl.int64)
    expert = rows // T
    token = rows - expert * T
    n = tl.load(M + expert, in_range, 0)
    live_row = in_range & (token < n)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < H
    base = rows64 * (2 * H)
    gate = tl.load(
        X + base[:, None] + cols[None, :],
        live_row[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        X + base[:, None] + H + cols[None, :],
        live_row[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # tl.sigmoid is illegal on one of the vendor compilers; the exp form is
    # the same function.
    val = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
    s = tl.load(S)
    out = _encode_fp8(tl.div_rn(val, s))
    # Rows past the expert's count load zeros and so encode to a zero byte,
    # which is what the reference leaves there.
    tl.store(
        Y + rows64[:, None] * H + cols[None, :],
        out,
        in_range[:, None] & col_mask[None, :],
    )


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    """Masked SiLU-and-mul quantized to E4M3FN with one per-tensor scale."""
    experts, tokens, width = input.shape
    half = width // 2
    out = torch.empty(
        (experts, tokens, half),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    if experts * tokens * half:
        if not input.is_contiguous():
            input = input.contiguous()
        if not masked_m.is_contiguous():
            masked_m = masked_m.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()
        # triton.next_power_of_2 and triton.cdiv are JIT functions in Triton
        # 3.9 and cost about 9 us each on the host; compute them in Python.
        block = 1 << (min(half, 1024) - 1).bit_length()
        rows = max(1, 2048 // block)
        total = experts * tokens
        _silu_quant.run(
            input,
            masked_m,
            scale,
            out.view(torch.uint8),
            tokens,
            half,
            total,
            rows,
            block,
            grid=((total + rows - 1) // rows, (half + block - 1) // block),
            warmup=False,
            num_warps=max(1, min(8, (rows * block) // 64)),
        )
    return out


solution = silu_and_mul_masked_post_per_tensor_quant

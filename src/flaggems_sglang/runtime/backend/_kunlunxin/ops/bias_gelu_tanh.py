# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Row bias, rounded to the input dtype, then tanh-approximate GELU --
Kunlun XPU specialization.

Same flat walk as the generic file with the two settings this chip is
charged by: it bills per tile iteration rather than per program, so the
tile is as wide as the backend's other elementwise kernels take it
(16384 elements against the generic 2048), and the block is chosen to
divide the element count so the loads and the store carry no mask, which
is what keeps the mask-free kernel separate from the one that needs it.
"""

import torch
import triton
import triton.language as tl

__all__ = ["bias_gelu_tanh"]

MAX_BLOCK = 16384
MIN_BLOCK = 1024
INT32_LIMIT = 1 << 31


@triton.jit
def _bias_gelu_tanh_exact(
    X,
    B,
    Y,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(
            0, BLOCK
        ).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs).to(tl.float32)
    bias = tl.load(B + offs % H).to(tl.float32)
    dtype = Y.dtype.element_ty
    z = (x + bias).to(dtype).to(tl.float32)
    # z * sigmoid(z * (2*sqrt(2/pi) + 2*sqrt(2/pi)*0.044715 * z^2)), the
    # sigmoid identity of 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715 z^3))).
    y = z * tl.sigmoid(z * (1.5957691216057308 + 0.07133681239524435 * z * z))
    tl.store(Y + offs, y.to(dtype))


@triton.jit
def _bias_gelu_tanh_masked(
    X,
    B,
    Y,
    total,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(
            0, BLOCK
        ).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # In range for every lane, including the lanes past the end.
    bias = tl.load(B + offs % H).to(tl.float32)
    dtype = Y.dtype.element_ty
    z = (x + bias).to(dtype).to(tl.float32)
    y = z * tl.sigmoid(z * (1.5957691216057308 + 0.07133681239524435 * z * z))
    tl.store(Y + offs, y.to(dtype), mask=mask)


def _pick_block(total):
    block = MAX_BLOCK
    while block > MIN_BLOCK and block >= 2 * total:
        block //= 2
    while block > MIN_BLOCK and total % block != 0:
        block //= 2
    return block, total % block == 0


def bias_gelu_tanh(x, bias):
    """`gelu((x + bias).to(x.dtype), approximate="tanh")`, as one kernel.

    `x` is a nonempty contiguous `[..., H]` fp16/bf16 tensor and `bias` is
    `[H]` of the same dtype and device. The result is a new tensor of the
    same shape, dtype and device; neither input is written.
    """
    src = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(src)
    h = x.shape[-1]
    total = out.numel()
    if total == 0 or h == 0:
        return out
    row = bias if bias.is_contiguous() else bias.contiguous()
    block, exact = _pick_block(total)
    wide = total >= INT32_LIMIT
    if exact:
        _bias_gelu_tanh_exact.run(
            src,
            row,
            out,
            h,
            block,
            wide,
            grid=(total // block,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    else:
        _bias_gelu_tanh_masked.run(
            src,
            row,
            out,
            total,
            h,
            block,
            wide,
            grid=((total + block - 1) // block,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return out


solution = bias_gelu_tanh

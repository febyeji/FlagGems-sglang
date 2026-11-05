# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Row bias, rounded to the input dtype, then tanh-approximate GELU --
general-purpose chip B specialization.

This is the generic file with one change: the row and the column are
kept apart instead of recovered from a flat index. The generic file
reads its bias at `offset % H`, which is an integer division per lane
even with H a compile-time constant; here a program takes a block of
rows by a block of columns, the bias index is the column vector
itself, and the element index is a scalar row base plus that same
vector. A program covers the same 2048 elements either way, so the
graded number says what the division is worth on this chip.
"""

import torch
import triton
import triton.language as tl

__all__ = ["bias_gelu_tanh"]

TILE = 2048
MAX_COLS = 2048
INT32_LIMIT = 1 << 31


@triton.jit
def _bias_gelu_tanh_kernel(
    X,
    B,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    CBLK: tl.constexpr,
    NC: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    # NC is 1 whenever a row fits in one tile, and both of these fold
    # away at compile time when it is.
    rb = pid // NC
    cb = pid % NC
    col = cb * CBLK + tl.arange(0, CBLK)
    r = rb * ROWS + tl.arange(0, ROWS)
    if WIDE:
        off = r.to(tl.int64)[:, None] * H + col[None, :]
    else:
        off = r[:, None] * H + col[None, :]
    dtype = Y.dtype.element_ty
    if EXACT:
        x = tl.load(X + off).to(tl.float32)
        bias = tl.load(B + col).to(tl.float32)
        z = (x + bias[None, :]).to(dtype).to(tl.float32)
        y = z * tl.sigmoid(
            z * (1.5957691216057308 + 0.07133681239524435 * z * z)
        )
        tl.store(Y + off, y.to(dtype))
    else:
        cmask = col < H
        mask = (r < M)[:, None] & cmask[None, :]
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(B + col, mask=cmask, other=0.0).to(tl.float32)
        z = (x + bias[None, :]).to(dtype).to(tl.float32)
        y = z * tl.sigmoid(
            z * (1.5957691216057308 + 0.07133681239524435 * z * z)
        )
        tl.store(Y + off, y.to(dtype), mask=mask)


def _shape(h, m):
    """Column block, column blocks per row, and rows per tile."""
    cols = 1
    while cols < h and cols < MAX_COLS:
        cols *= 2
    nc = (h + cols - 1) // cols
    rows = 1 if nc > 1 else max(1, TILE // cols)
    while rows > 1 and rows > m:
        rows //= 2
    return cols, nc, rows


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
    m = total // h
    cols, nc, rows = _shape(h, m)
    _bias_gelu_tanh_kernel.run(
        src,
        row,
        out,
        m,
        h,
        cols,
        nc,
        rows,
        h % cols == 0 and m % rows == 0,
        total >= INT32_LIMIT,
        grid=(((m + rows - 1) // rows) * nc,),
        warmup=False,
        num_warps=1 if total <= 8192 else 4,
        num_stages=1,
    )
    return out


solution = bias_gelu_tanh

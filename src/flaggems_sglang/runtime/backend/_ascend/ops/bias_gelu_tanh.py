# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Row bias, rounded to the input dtype, then tanh-approximate GELU --
Ascend NPU specialization.

The generic file walks a flat element index and recovers the bias column
with a modulo. This one keeps the row and the column apart instead: a
program takes a block of rows by a block of columns, so the bias is a
plain `tl.arange` read once per tile and the element index is a scalar
row base plus that same column vector. A fixed number of programs walk
the tiles in a loop whose trip count is a compile-time constant, which
is the shape this chip's compiler schedules best.
"""

import torch
import triton
import triton.language as tl

__all__ = ["bias_gelu_tanh"]

# One program per vector core; the tiles are spread over them in a loop
# rather than over a grid as wide as the tensor.
PROGRAMS = 40
# Elements a tile covers, and the ceiling on its column side.
TILE = 4096
MAX_COLS = 1024
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
    WORK: tl.constexpr,
    TRIPS: tl.constexpr,
    PROGS: tl.constexpr,
    GUARD: tl.constexpr,
    EXACT: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, CBLK)
    rows = tl.arange(0, ROWS)
    dtype = Y.dtype.element_ty
    for i in range(TRIPS):
        w = pid + i * PROGS
        # The trip count covers every program, so the last pass runs
        # short unless the work divides evenly over them.
        if (not GUARD) or w < WORK:
            # NC is 1 whenever a row fits in one tile, and both of these
            # fold away at compile time when it is.
            rb = w // NC
            cb = w % NC
            col = cb * CBLK + cols
            r = rb * ROWS + rows
            if WIDE:
                off = r.to(tl.int64)[:, None] * H + col[None, :]
            else:
                off = r[:, None] * H + col[None, :]
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


def _launch(src, row, out, total, m, h):
    """Shape the grid and enter the kernel.

    The program count is fixed and the trip count is a compile-time
    constant, so the kernel's loop unrolls and the grid never depends on
    a value the host has to divide at launch time.
    """
    cols, nc, rows = _shape(h, m)
    nrb = (m + rows - 1) // rows
    work = nrb * nc
    progs = min(work, PROGRAMS)
    trips = (work + progs - 1) // progs
    _bias_gelu_tanh_kernel.run(
        src,
        row,
        out,
        m,
        h,
        cols,
        nc,
        rows,
        work,
        trips,
        progs,
        trips * progs != work,
        h % cols == 0 and m % rows == 0,
        total >= INT32_LIMIT,
        grid=(progs,),
        warmup=False,
        num_warps=1,
        num_stages=1,
    )


def bias_gelu_tanh(x, bias):
    """`gelu((x + bias).to(x.dtype), approximate="tanh")`, as one kernel.

    `x` is a nonempty contiguous `[..., H]` fp16/bf16 tensor and `bias` is
    `[H]` of the same dtype and device. The result is a new tensor of the
    same shape, dtype and device; neither input is written. The call runs
    on the inputs' device whichever device is current.
    """
    src = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(src)
    h = x.shape[-1]
    total = out.numel()
    if total == 0 or h == 0:
        return out
    row = bias if bias.is_contiguous() else bias.contiguous()
    # Triton launches on the device that is current and rejects a
    # pointer it cannot reach from there, so an input that lives
    # elsewhere has to make its own device current first. This backend
    # reaches torch as `npu`, so the module comes from the tensor.
    device_fn = torch.cuda if src.is_cuda else getattr(torch, src.device.type)
    if device_fn.current_device() == src.get_device():
        _launch(src, row, out, total, total // h, h)
    else:
        with device_fn.device(src.device):
            _launch(src, row, out, total, total // h, h)
    return out


solution = bias_gelu_tanh

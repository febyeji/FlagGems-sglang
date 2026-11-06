# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Row bias, rounded to the input dtype, then tanh-approximate GELU.

The operator broadcasts a `[H]` bias over the last dimension of a
contiguous `[..., H]` tensor, rounds the sum back to the input dtype and
applies the tanh approximation of GELU. One pass over memory reads two
bytes and writes two bytes per element, where the reference materializes
an fp32 sum, an fp32 bias, a rounded copy and the GELU output in turn.

The rounding between the add and the GELU is part of the definition, not
an artifact: the reference writes `(x.float() + bias.float()).to(dtype)`
and feeds that to GELU, so a kernel that carries the fp32 sum straight
into the activation computes a different function.

Anything that does not saturate the bus is paid for on the host: on one
RTX 3090 a call costs about 33 microseconds before the first byte moves,
which is every shape below roughly 100 MB. The launcher therefore keeps
its own arithmetic in plain Python, hands the kernel positional
arguments, keeps the mask and the element count out of the signature on
the common path, and holds the compiled kernel so that the common path
enters the launcher without rebinding its arguments first.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.runtime import driver

__all__ = ["bias_gelu_tanh"]

# The launcher halves the block its size class wants until the block
# divides the element count and the mask-free kernel applies. It stops
# at MIN_BLOCK rather than shrinking the tile for one lane's sake.
MIN_BLOCK = 256
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
    # One flat block of elements, with the block dividing the count so
    # every lane is in range. The last dimension is the fastest, so the
    # bias column of an element is its flat offset modulo H, and H is a
    # compile-time constant that turns the modulo into a multiply.
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
    # 0.5 * z * (1 + tanh(u)) is z * sigmoid(2u), which spends one
    # exponential instead of a tanh and a half-sum. Folding the 2 into
    # the polynomial leaves z * sigmoid(z * (A + B * z^2)) with
    # A = 2*sqrt(2/pi) and B = A*0.044715. It saturates at both ends on
    # its own: a large positive z gives sigmoid 1 and returns z, a large
    # negative one gives sigmoid 0 and returns zero, with no clamp.
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
    # The same walk for an element count no block of at least MIN_BLOCK
    # divides.
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(
            0, BLOCK
        ).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # The column stays inside [0, H) for every lane, including the lanes
    # past the end of the tensor, so this load is in range unmasked.
    bias = tl.load(B + offs % H).to(tl.float32)
    dtype = Y.dtype.element_ty
    z = (x + bias).to(dtype).to(tl.float32)
    y = z * tl.sigmoid(z * (1.5957691216057308 + 0.07133681239524435 * z * z))
    tl.store(Y + offs, y.to(dtype), mask=mask)


def _config(total):
    """Block and warp count for a size class, fitted to `total`.

    The streaming sizes want the widest tile the backend schedules well.
    A shape that fits in a wave or two never reaches the bus at all, so
    it takes a narrow single-warp block and is done.
    """
    if total >= 1 << 20:
        block, warps = 2048, 4
    elif total <= 8192:
        block, warps = 256, 1
    else:
        block, warps = 1024, 4
    while block > MIN_BLOCK and total % block != 0:
        block //= 2
    return block, warps, total % block == 0


@functools.lru_cache(maxsize=None)
def _compiled(dtype, h, block, warps, wide, device):
    """The compiled exact kernel for one specialization, or None.

    `JITFunction.run` rebinds the arguments and rebuilds the
    specialization key on every call before it reaches the launcher,
    which is 14 of the 38 microseconds a small call takes on one RTX
    3090. It hands back the compiled kernel, and that kernel can be
    entered directly, so the launcher keeps one per specialization.

    Everything the kernel is specialized on is in the key: the dtype,
    the three compile-time constants, the warp count and the device.
    The one dimension that is not an argument here is whether Triton
    may assume 16-byte aligned pointers, which the caller settles by
    taking this path only for pointers that are aligned, as the probes
    below are. Nothing about a call's data is kept — this is the
    compiled kernel, the same object Triton's own cache holds.

    A backend whose `run` does not hand the kernel back returns None
    here and keeps the ordinary path.
    """
    probe = torch.empty(block, dtype=dtype, device=device)
    row = torch.empty(h, dtype=dtype, device=device)
    kernel = _bias_gelu_tanh_exact.run(
        probe,
        row,
        probe,
        h,
        block,
        wide,
        grid=(1,),
        warmup=False,
        num_warps=warps,
        num_stages=1,
    )
    if kernel is None or not hasattr(kernel, "__getitem__"):
        return None
    return kernel


def _launch(src, row, out, h, total, block, warps, exact, wide):
    """Enter the kernel through Triton.

    `triton.cdiv` and `triton.next_power_of_2` are JIT callables that
    cost about 9 microseconds a call, which is most of the budget of a
    small shape, so the grid is counted out in plain Python.
    """
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
            num_warps=warps,
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
            num_warps=warps,
            num_stages=1,
        )


def bias_gelu_tanh(x, bias):
    """`gelu((x + bias).to(x.dtype), approximate="tanh")`, as one kernel.

    `x` is a nonempty contiguous `[..., H]` fp16/bf16 tensor and `bias` is
    `[H]` of the same dtype and device. The result is a new tensor of the
    same shape, dtype and device; neither input is written. The call runs
    on the inputs' device whichever device is current.
    """
    # `empty_like` of a contiguous source is itself contiguous, which is
    # what the flat index the kernel walks needs.
    src = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(src)
    h = x.shape[-1]
    total = out.numel()
    if total == 0 or h == 0:
        return out
    row = bias if bias.is_contiguous() else bias.contiguous()
    block, warps, exact = _config(total)
    wide = total >= INT32_LIMIT
    # Triton launches on the device that is current and rejects a
    # pointer it cannot reach from there, so an input that lives
    # elsewhere has to make its own device current first. Asking costs
    # about 1.5 microseconds and entering the guard about 2.6, which is
    # a difference worth having on a path that is nearly all host work,
    # and `is_cuda` keeps the question off the device object. Every
    # vendor here reaches torch as `cuda` except Ascend, which reaches
    # it as `npu`.
    dev = src.device
    device_fn = torch.cuda if src.is_cuda else getattr(torch, dev.type)
    if device_fn.current_device() != dev.index:
        with device_fn.device(dev):
            _launch(src, row, out, h, total, block, warps, exact, wide)
        return out
    # The held kernel takes the mask-free path with aligned pointers,
    # which is every shape whose element count a block divides. The
    # output is a fresh allocation and so always aligned; the inputs
    # are the caller's and may be a view that is not.
    if exact and ((src.data_ptr() | row.data_ptr()) & 15) == 0:
        kernel = _compiled(src.dtype, h, block, warps, wide, dev)
        if kernel is not None:
            kernel[(total // block, 1, 1)](
                src,
                row,
                out,
                stream=driver.active.get_current_stream(dev.index),
            )
            return out
    _launch(src, row, out, h, total, block, warps, exact, wide)
    return out


solution = bias_gelu_tanh

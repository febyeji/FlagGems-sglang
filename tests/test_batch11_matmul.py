# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, backend="generic"):
    directory = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    spec = importlib.util.spec_from_file_location(
        name + backend, ROOT / f"src/flaggems_sglang/{directory}/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def unpack(t):
    shifts = torch.arange(8, device=t.device) * 4
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=t.device)
    return (
        ((t[..., None].to(torch.int64) >> shifts) & 15).index_select(-1, order)
    ).reshape(t.shape[0], -1)


def awq_inputs(m, n, k, g, dtype, strided=False):
    def tensor(shape, packed=False):
        physical = (shape[0] + 2, shape[1] * 2 + 4) if strided else shape
        if packed:
            t = torch.randint(
                -(2**31), 2**31 - 1, physical, device="cuda", dtype=torch.int32
            )
        else:
            t = torch.randn(physical, device="cuda", dtype=dtype)
        return t[1:-1, 2:-2:2] if strided else t

    a = tensor((m, k))
    w = tensor((k, n // 8), True)
    z = tensor((k // g, n // 8), True)
    s = tensor((k // g, n))
    s.mul_(0.1)
    return a, w, s, z


def awq_reference(a, w, s, z):
    group = w.shape[0] // s.shape[0]
    weight = (unpack(w) - unpack(z).repeat_interleave(group, 0)).to(
        s.dtype
    ) * s.repeat_interleave(group, 0)
    return a @ weight.to(a.dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
@pytest.mark.parametrize("split", [1, 3, 8])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "m,n,k,g",
    [
        (1, 24, 64, 16),
        (3, 128, 96, 32),
        (17, 136, 160, 40),
        (64, 256, 512, 128),
        (129, 512, 1024, 128),
    ],
)
def test_awq_gemm(dtype, backend, split, strided, m, n, k, g):
    torch.manual_seed(162023)
    torch.backends.cuda.matmul.allow_tf32 = False
    args = awq_inputs(m, n, k, g, dtype, strided)
    expected = awq_reference(*args)
    actual = load("awq_gemm", backend)(*args, split)
    assert actual.shape == (m, n) and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("split", [1, 7, 64])
@pytest.mark.parametrize("backend", ["generic", "ascend"])
def test_awq_gemm_short_k(split, backend):
    torch.manual_seed(162024)
    args = awq_inputs(2, 40, 15, 5, torch.float16, True)
    actual = load("awq_gemm", backend)(*args, split)
    torch.testing.assert_close(
        actual, awq_reference(*args), atol=0.1, rtol=0.1
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize(
    "n,k,g",
    [
        (24, 64, 16),
        (40, 15, 5),
        (136, 160, 40),
        (136, 256, 128),
        (512, 1024, 128),
        (24, 2, 1),
        (24, 1024, 512),
        (24, 1026, 513),
    ],
)
def test_awq_gemm_xpu_dequant_layout(dtype, strided, wide, n, k, g):
    torch.manual_seed(1621027)
    _, w, s, z = awq_inputs(1, n, k, g, dtype, strided)
    expected = (unpack(w) - unpack(z).repeat_interleave(g, 0)).to(
        dtype
    ) * s.repeat_interleave(g, 0)
    decoded = torch.empty((n, k), device="cuda", dtype=dtype)
    kernel = load("awq_gemm", "kunlunxin").__globals__["_dequantize"]
    block = min(8192, max(1024, 1 << (max(1, k * n // 24) - 1).bit_length()))
    grid = (k * n + block - 1) // block
    kernel[(grid,)](
        w,
        s,
        z,
        decoded,
        k,
        n,
        g,
        *w.stride(),
        *s.stride(),
        *z.stride(),
        wide,
        block,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    torch.testing.assert_close(decoded, expected.t(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("group", [1, 40, 128, 256, 512, 513])
@pytest.mark.parametrize("strided", [False, True])
def test_awq_gemm_xpu_decode_groups(dtype, group, strided):
    torch.manual_seed(1621028)
    torch.backends.cuda.matmul.allow_tf32 = False
    args = awq_inputs(3, 24, group * 2, group, dtype, strided)
    actual = load("awq_gemm", "kunlunxin")(*args, 3)
    torch.testing.assert_close(
        actual, awq_reference(*args), atol=0.1, rtol=0.1
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("split", [1, 3, 8])
@pytest.mark.parametrize(
    "m,n,k,g",
    [
        (1, 136, 8192, 128),
        (4, 40, 15, 5),
        (4, 136, 1026, 513),
        (1, 24, 160, 40),
        (2, 512, 4096, 128),
        (5, 136, 1024, 128),
    ],
)
def test_awq_gemm_ascend_small_rows(dtype, strided, split, m, n, k, g):
    torch.manual_seed(1622027)
    torch.backends.cuda.matmul.allow_tf32 = False
    a, w, s, z = awq_inputs(m, n, k, g, dtype, strided)
    weight = (unpack(w) - unpack(z).repeat_interleave(g, 0)).to(
        s.dtype
    ) * s.repeat_interleave(g, 0)
    expected = (a.float() @ weight.float()).to(dtype)
    actual = load("awq_gemm", "ascend")(a, w, s, z, split)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)


def mxfp8_reference(a, sa, b, sb, dtype):
    a_deq = a.float() * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    b_deq = b.float() * torch.exp2(sb.float() - 127).repeat_interleave(32, 1)
    return (a_deq @ b_deq.t()).to(dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
@pytest.mark.parametrize(
    "m,n,k", [(1, 7, 32), (17, 67, 96), (64, 128, 256), (129, 257, 512)]
)
def test_mxfp8_matmul(dtype, strided, backend, m, n, k):
    torch.manual_seed(164023)
    torch.backends.cuda.matmul.allow_tf32 = False

    def make(rows):
        raw = (torch.randn((rows, k), device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        )
        scale = torch.randint(
            124, 130, (rows, k // 32), device="cuda", dtype=torch.uint8
        )
        if strided:
            raw_store = torch.zeros(
                (rows + 2, k * 2 + 4), device="cuda", dtype=torch.uint8
            )
            raw_store[1:-1, 2:-2:2] = raw.view(torch.uint8)
            raw = raw_store[1:-1, 2:-2:2].view(raw.dtype)
            scale_store = torch.zeros(
                (rows + 2, (k // 32) * 2 + 4), device="cuda", dtype=torch.uint8
            )
            scale_store[1:-1, 2:-2:2] = scale
            scale = scale_store[1:-1, 2:-2:2]
        return raw, scale

    a, sa = make(m)
    b, sb = make(n)
    actual = load("mxfp8_block_scaled_matmul", backend)(a, sa, b, sb, dtype)
    expected = mxfp8_reference(a, sa, b, sb, dtype)
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.01)


@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
def test_mxfp8_all_encoding_bytes(backend):
    a = (
        torch.arange(256, device="cuda", dtype=torch.int32)
        .to(torch.uint8)[:, None]
        .repeat(1, 32)
        .view(torch.float8_e4m3fn)
    )
    b = torch.ones((1, 32), device="cuda").to(a.dtype)
    sa = torch.full((256, 1), 127, device="cuda", dtype=torch.uint8)
    sb = torch.full((1, 1), 127, device="cuda", dtype=torch.uint8)
    actual = load("mxfp8_block_scaled_matmul", backend)(
        a, sa, b, sb, torch.float32
    )
    expected = mxfp8_reference(a, sa, b, sb, torch.float32)
    torch.testing.assert_close(
        actual, expected, atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.parametrize("exponent", [0, 1, 9, 10, 246, 247, 254, 255])
@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
def test_mxfp8_scale_range(exponent, backend):
    a = torch.ones((3, 32), device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones((7, 32), device="cuda").to(torch.float8_e4m3fn)
    sa = torch.full((3, 1), exponent, device="cuda", dtype=torch.uint8)
    sb = torch.full(
        (7, 1), 254 - min(exponent, 254), device="cuda", dtype=torch.uint8
    )
    actual = load("mxfp8_block_scaled_matmul", backend)(
        a, sa, b, sb, torch.float32
    )
    expected = mxfp8_reference(a, sa, b, sb, torch.float32)
    torch.testing.assert_close(
        actual, expected, atol=0.5, rtol=0.01, equal_nan=True
    )


@pytest.mark.parametrize("k", [4096, 8192])
@pytest.mark.parametrize("split", [1, 8])
def test_awq_gemm_xpu_long_reduction(k, split):
    torch.manual_seed(162028)
    args = awq_inputs(9, 24, k, 128, torch.float16, True)
    actual = load("awq_gemm", "kunlunxin")(*args, split)
    torch.testing.assert_close(
        actual, awq_reference(*args), atol=0.1, rtol=0.1
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("k", [4096, 8192])
@pytest.mark.parametrize("split", [1, 3, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_awq_gemm_xpu_single_row(dtype, k, split, strided):
    torch.manual_seed(162026)
    torch.backends.cuda.matmul.allow_tf32 = False
    a, w, s, z = awq_inputs(1, 136, k, 128, dtype, strided)
    weight = (unpack(w) - unpack(z).repeat_interleave(128, 0)).to(
        s.dtype
    ) * s.repeat_interleave(128, 0)
    # Use full accumulation to avoid cuBLAS BF16 partial-sum rounding.
    expected = (a.float() @ weight.float()).to(dtype)
    actual = load("awq_gemm", "kunlunxin")(a, w, s, z, split)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("k", [4096, 8192])
def test_mxfp8_xpu_long_reduction(k):
    torch.manual_seed(164028)
    a = (torch.randn((9, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((17, k), device="cuda") * 0.5).to(a.dtype)
    sa = torch.randint(
        124, 130, (9, k // 32), device="cuda", dtype=torch.uint8
    )
    sb = torch.randint(
        124, 130, (17, k // 32), device="cuda", dtype=torch.uint8
    )
    actual = load("mxfp8_block_scaled_matmul", "kunlunxin")(
        a, sa, b, sb, torch.bfloat16
    )
    expected = mxfp8_reference(a, sa, b, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.01)

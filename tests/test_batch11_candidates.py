# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
FP8 = [torch.float8_e4m3fn, torch.float8_e5m2]


def load(name, backend="generic"):
    path = ROOT / "src/flaggems_sglang"
    path /= (
        f"ops/{name}.py"
        if backend == "generic"
        else f"runtime/backend/_{backend}/ops/{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def make(shape, dtype, strided=False):
    if strided:
        return torch.randn(
            (*shape[:-1], shape[-1] * 2), device="cuda", dtype=dtype
        )[..., ::2]
    return torch.randn(shape, device="cuda", dtype=dtype)


def fp8_reference(x, scale, dtype):
    maximum = torch.finfo(dtype).max
    return (x.float() * scale).clamp(-maximum, maximum).to(dtype)


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )


def unpack(t):
    shifts = torch.arange(8, device=t.device) * 4
    fields = (t[..., None].to(torch.int64) >> shifts) & 15
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=t.device)
    return fields.index_select(-1, order).reshape(t.shape[0], -1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "k,n,g",
    [
        (1, 8, 1),
        (32, 24, 32),
        (128, 128, 32),
        (256, 768, 64),
        (256, 2048, 128),
        (512, 4096, 512),
    ],
)
@pytest.mark.parametrize("backend", ["generic", "ascend", "kunlunxin"])
def test_awq(dtype, strided, k, n, g, backend):
    torch.manual_seed(161022)

    def packed(shape):
        if strided:
            return torch.randint(
                -(2**31),
                2**31 - 1,
                (shape[0], shape[1] * 2),
                device="cuda",
                dtype=torch.int32,
            )[:, ::2]
        return torch.randint(
            -(2**31), 2**31 - 1, shape, device="cuda", dtype=torch.int32
        )

    w, z = packed((k, n // 8)), packed((k // g, n // 8))
    s = make((k // g, n), dtype, strided)
    expected = (unpack(w) - unpack(z).repeat_interleave(g, 0)).to(
        dtype
    ) * s.repeat_interleave(g, 0)
    actual = load("awq_dequantize", backend)(w, s, z)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("backend", ["generic", "ascend", "kunlunxin"])
@pytest.mark.parametrize(
    "k,n,g",
    [
        (96, 1032, 3),
        (35, 1032, 7),
        (68, 520, 17),
        (128, 264, 32),
        (512, 2056, 128),
        (1026, 1032, 513),
    ],
)
def test_awq_group_tails(dtype, strided, backend, k, n, g):
    torch.manual_seed(161029)

    def packed(rows):
        storage = torch.randint(
            -(2**31),
            2**31 - 1,
            (rows + 2, n // 8 * (2 if strided else 1) + 4),
            device="cuda",
            dtype=torch.int32,
        )
        return storage[1:-1, 2:-2:2] if strided else storage[1:-1, 2:-2]

    w, z = packed(k), packed(k // g)
    s = make((k // g, n), dtype, strided)
    expected = (unpack(w) - unpack(z).repeat_interleave(g, 0)).to(
        dtype
    ) * s.repeat_interleave(g, 0)
    before = [t.clone() for t in (w, s, z)]
    actual = load("awq_dequantize", backend)(w, s, z)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for original, unchanged in zip((w, s, z), before):
        torch.testing.assert_close(original, unchanged, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
@pytest.mark.parametrize("dtype", DTYPES[:2])
@pytest.mark.parametrize("out_dtype", FP8)
@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 1.702, -3.0, 64.0])
def test_fp8_all_input_bits(backend, dtype, out_dtype, scale):
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
    )
    actual = load("fp8_quantize", backend)(x, scale, out_dtype)
    assert_bytes(actual, fp8_reference(x, scale, out_dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", FP8)
@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
@pytest.mark.parametrize(
    "shape", [(1,), (3, 128), (2, 3, 256), (257, 2048), (0, 128)]
)
def test_fp8_sliced(backend, dtype, out_dtype, shape):
    torch.manual_seed(163022)
    full = make((*shape[:-1], shape[-1] + 17), dtype)
    x = full[..., 17:]
    before = x.clone()
    actual = load("fp8_quantize", backend)(x, 1.75, out_dtype)
    assert actual.is_contiguous()
    assert_bytes(actual, fp8_reference(x, 1.75, out_dtype))
    torch.testing.assert_close(x, before, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
@pytest.mark.parametrize("out_dtype", FP8)
def test_fp8_rounding_boundaries(backend, out_dtype):
    # Probe both sides of every finite FP8 rounding midpoint in FP32.
    values = (
        torch.arange(128, device="cuda", dtype=torch.uint8)
        .view(out_dtype)
        .float()
    )
    values = values[torch.isfinite(values)]
    midpoint = (values[:-1] + values[1:]) * 0.5
    toward_zero = torch.nextafter(midpoint, torch.zeros_like(midpoint))
    toward_inf = torch.nextafter(
        midpoint, torch.full_like(midpoint, float("inf"))
    )
    positive = torch.cat((values, midpoint, toward_zero, toward_inf))
    special = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")], device="cuda"
    )
    x = torch.cat((positive, -positive, special))
    actual = load("fp8_quantize", backend)(x, 1.0, out_dtype)
    assert_bytes(actual, fp8_reference(x, 1.0, out_dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("count", [1, 31, 4095, 4096, 4097, 12345, 4194321])
@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
def test_absmax(dtype, strided, count, backend):
    torch.manual_seed(165022)
    x = make((count,), dtype, strided)
    scale = torch.zeros((1,), device="cuda")
    actual = load("per_tensor_absmax_fp8", backend)(x, scale)
    expected = (x.float().abs().amax() / 448).reshape(1)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    actual = load("per_tensor_absmax_fp8", backend)(x, scale)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "shape", [(1,), (17, 31), (3, 1024), (13, 769), (257, 4096)]
)
@pytest.mark.parametrize("backend", ["generic", "ascend", "kunlunxin"])
def test_dynamic_quant(dtype, static, strided, shape, backend):
    torch.manual_seed(166022)
    x = make(shape, dtype, strided)
    q = torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)
    s = torch.tensor([0.037], device="cuda")
    expected_s = (
        s.clone() if static else (x.float().abs().amax() / 448).reshape(1)
    )
    expected_q = (x.float() / expected_s).clamp(-448, 448).to(q.dtype)
    actual_q, actual_s = load("per_tensor_quant_fp8", backend)(x, q, s, static)
    torch.testing.assert_close(actual_s, expected_s, atol=1e-7, rtol=1e-6)
    # Check FP8 rounding by comparing the encoded bytes.
    assert_bytes(actual_q, expected_q)


@pytest.mark.parametrize("backend", ["generic", "ascend", "kunlunxin"])
@pytest.mark.parametrize("count", [1, 1024, 1025, 8192, 8193])
def test_dynamic_zero(count, backend):
    x = torch.zeros(count, device="cuda")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.zeros(1, device="cuda")
    actual, scale = load("per_tensor_quant_fp8", backend)(x, q, s, False)
    assert scale.item() == 0
    assert torch.isnan(actual.float()).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "count",
    [
        255,
        256,
        257,
        1023,
        1025,
        2049,
        4095,
        4097,
        8191,
        8192,
        8193,
        16383,
        16384,
        16385,
    ],
)
@pytest.mark.parametrize("backend", ["generic", "ascend", "kunlunxin"])
def test_dynamic_reduction_boundaries(dtype, strided, count, backend):
    torch.manual_seed(166023)
    x = make((count,), dtype, strided)
    storage = torch.full(
        (count * 3 + 8,), 19, device="cuda", dtype=torch.uint8
    )
    q = storage[4:-4:3].view(torch.float8_e4m3fn)
    s = torch.tensor([123.0], device="cuda")
    op = load("per_tensor_quant_fp8", backend)
    for position, maximum in [
        (0, 100.0),
        (count - 1, 200.0),
        (count // 2, 300.0),
    ]:
        x[position] = -maximum
        expected_s = (x.float().abs().amax() / 448).reshape(1)
        expected_q = (x.float() / expected_s).clamp(-448, 448).to(q.dtype)
        actual_q, actual_s = op(x, q, s, False)
        assert actual_q.data_ptr() == q.data_ptr()
        assert actual_s.data_ptr() == s.data_ptr()
        assert_bytes(actual_q, expected_q)
        torch.testing.assert_close(actual_s, expected_s, atol=1e-7, rtol=1e-6)
        assert (storage[:4] == 19).all() and (storage[-4:] == 19).all()
        assert (storage[5:-4:3] == 19).all() and (storage[6:-4:3] == 19).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("count", [4097, 131073, 4194321])
@pytest.mark.parametrize("backend", ["generic", "kunlunxin", "ascend"])
def test_absmax_reused_output(dtype, count, backend):
    torch.manual_seed(165023)
    x = make((count,), dtype, True)
    s = torch.zeros(1, device="cuda")
    op = load("per_tensor_absmax_fp8", backend)
    for position, maximum in [(0, 1000.0), (count - 1, 10.0)]:
        x.fill_(0.25)
        x[position] = -maximum
        s.zero_()
        actual = op(x, s)
        assert actual.data_ptr() == s.data_ptr()
        expected = (x.float().abs().amax() / 448).reshape(1)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("backend", ["generic", "ascend", "kunlunxin"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.037, 1.0, 3.25])
def test_dynamic_static_all_bits(backend, dtype, scale):
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
    )
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.tensor([scale], device="cuda")
    before = s.clone()
    load("per_tensor_quant_fp8", backend)(x, q, s, True)
    expected = (x.float() / before).clamp(-448, 448).to(q.dtype)
    assert_bytes(q, expected)
    torch.testing.assert_close(s, before, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["ascend", "kunlunxin"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("count", [163841, 1048577])
@pytest.mark.parametrize("offset", [4, 5])
@pytest.mark.parametrize("static", [False, True])
def test_dynamic_large_output_view(backend, dtype, count, offset, static):
    torch.manual_seed(166026)
    x = make((count,), dtype, True)
    storage = torch.full(
        (count + offset + 4,), 19, device="cuda", dtype=torch.uint8
    )
    q = storage[offset:-4].view(torch.float8_e4m3fn)
    s = torch.tensor([0.037], device="cuda")
    expected_s = (
        s.clone() if static else (x.float().abs().amax() / 448).reshape(1)
    )
    expected_q = (x.float() / expected_s).clamp(-448, 448).to(q.dtype)
    aq, actual_s = load("per_tensor_quant_fp8", backend)(x, q, s, static)
    assert (
        aq.data_ptr() == q.data_ptr() and actual_s.data_ptr() == s.data_ptr()
    )
    assert_bytes(aq, expected_q)
    torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)
    assert (storage[:offset] == 19).all() and (storage[-4:] == 19).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "count", [32769, 65537, 131071, 131072, 131073, 262143, 262144, 262145]
)
def test_dynamic_medium_reduction(dtype, strided, count):
    torch.manual_seed(166028)
    x = make((count,), dtype, strided)
    storage = torch.full(
        (count * 3 + 8,), 19, device="cuda", dtype=torch.uint8
    )
    q = storage[4:-4:3].view(torch.float8_e4m3fn)
    s = torch.tensor([123.0], device="cuda")
    op = load("per_tensor_quant_fp8")
    for position, maximum in [(0, 100.0), (count - 1, 10.0)]:
        x.fill_(0.25)
        x[position] = -maximum
        expected_s = (x.float().abs().amax() / 448).reshape(1)
        expected_q = (x.float() / expected_s).clamp(-448, 448).to(q.dtype)
        actual_q, actual_s = op(x, q, s, False)
        assert actual_q.data_ptr() == q.data_ptr()
        assert actual_s.data_ptr() == s.data_ptr()
        assert_bytes(actual_q, expected_q)
        torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)
        assert (storage[:4] == 19).all() and (storage[-4:] == 19).all()
        assert (storage[5:-4:3] == 19).all() and (storage[6:-4:3] == 19).all()
    x.zero_()
    op(x, q, s, False)
    assert s.item() == 0
    assert torch.isnan(q.float()).all()

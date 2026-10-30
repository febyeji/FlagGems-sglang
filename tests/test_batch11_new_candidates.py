# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["generic", "ascend", "kunlunxin"]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def load(name, backend):
    path = ROOT / "src/flaggems_sglang"
    path /= (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    spec = importlib.util.spec_from_file_location(
        name + backend, path / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def make(shape, dtype, strided=False):
    shape = tuple(shape)
    full = torch.randn(
        (*shape[:-1], shape[-1] * (2 if strided else 1)),
        device="cuda",
        dtype=dtype,
    )
    return full[..., ::2] if strided else full


def group_reference(x, group, eps=1e-10, lower=-448.0, upper=448.0):
    values = x.float().reshape(-1, group)
    scale = values.abs().amax(-1, keepdim=True).clamp(min=eps) / upper
    q = (values / scale).clamp(lower, upper).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), scale.reshape(
        x.shape[:-1] + (x.shape[-1] // group,)
    )


def assert_fp8(q, expected):
    assert q.dtype == expected.dtype and q.shape == expected.shape
    torch.testing.assert_close(
        q.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("backend", BACKENDS + ["iluvatar"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "shape,group",
    [
        ((1, 32), 32),
        ((3, 128), 64),
        ((5, 768), 128),
        ((17, 1024), 256),
        ((2, 3, 2048), 512),
        ((161, 4096), 128),
        ((7, 39), 3),
    ],
)
def test_group_fp8(backend, dtype, strided, shape, group):
    torch.manual_seed(168023)
    x = make(shape, dtype, strided)
    if shape[-1] > group:
        x[..., :group].zero_()
    expected_q, expected_s = group_reference(x, group)
    q, scale = load("per_token_group_quant_fp8", backend)(x, group)
    assert_fp8(q, expected_q)
    torch.testing.assert_close(scale, expected_s, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS + ["iluvatar"])
@pytest.mark.parametrize("dtype", DTYPES[:2])
@pytest.mark.parametrize("group", [32, 128])
def test_group_fp8_all_bits(backend, dtype, group):
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
        .reshape(-1, 256)
    )
    expected_q, expected_s = group_reference(x, group)
    q, scale = load("per_token_group_quant_fp8", backend)(x, group)
    assert_fp8(q, expected_q)
    torch.testing.assert_close(
        scale, expected_s, atol=1e-7, rtol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize("backend", BACKENDS + ["iluvatar"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "tokens,hidden,group",
    [
        (1, 128, 128),
        (5, 768, 128),
        (161, 4096, 128),
        (3, 39, 3),
        (17, 512, 64),
        (5, 1024, 256),
    ],
)
@pytest.mark.parametrize("eps,limit", [(1e-10, 448.0), (1e-3, 224.0)])
def test_group_v2(backend, strided, tokens, hidden, group, eps, limit):
    torch.manual_seed(167023)
    x = make((tokens, hidden), torch.bfloat16, strided)
    if tokens > 1:
        x[0].zero_()
    step = 3 if strided else 1
    raw = torch.full(
        (x.numel() * step + 8,), 19, device="cuda", dtype=torch.uint8
    )
    q = raw[4:-4:step].reshape(x.shape).view(torch.float8_e4m3fn)
    storage_s = torch.full(
        (x.numel() // group * step + 8,), -99.0, device="cuda"
    )
    scale = storage_s[4:-4:step].reshape(tokens, hidden // group)
    expected_q, expected_s = group_reference(x, group, eps, -limit, limit)
    result = load("per_token_group_quant_8bit_v2", backend)(
        x, q, scale, group, eps, -limit, limit
    )
    assert (
        result[0].data_ptr() == q.data_ptr()
        and result[1].data_ptr() == scale.data_ptr()
    )
    assert_fp8(q, expected_q)
    torch.testing.assert_close(scale, expected_s, atol=1e-7, rtol=1e-6)
    assert (raw[:4] == 19).all() and (raw[-4:] == 19).all()
    assert (storage_s[:4] == -99).all() and (storage_s[-4:] == -99).all()
    if strided:
        assert (raw[5:-4:3] == 19).all() and (raw[6:-4:3] == 19).all()
        assert (storage_s[5:-4:3] == -99).all() and (
            storage_s[6:-4:3] == -99
        ).all()


@pytest.mark.parametrize("backend", BACKENDS + ["iluvatar"])
@pytest.mark.parametrize("dtype", DTYPES[:2])
@pytest.mark.parametrize("group", [32, 128, 256])
def test_group_v2_all_bits(backend, dtype, group):
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
        .reshape(-1, 256)
    )
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((x.shape[0], 256 // group), device="cuda")
    expected_q, expected_s = group_reference(x, group)
    load("per_token_group_quant_8bit_v2", backend)(
        x, q, scale, group, 1e-10, -448.0, 448.0
    )
    assert_fp8(q, expected_q)
    torch.testing.assert_close(
        scale, expected_s, atol=1e-7, rtol=1e-6, equal_nan=True
    )


def int8_reference(x, scale_dtype, cal_sum):
    value = x.float()
    maximum = value.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    scaled = value * (127 / maximum)
    q = (scaled.sign() * (scaled.abs() + 0.5).floor()).to(torch.int8)
    scale = (maximum / 127).to(scale_dtype)
    return (q, scale, value.sum(-1).to(x.dtype)) if cal_sum else (q, scale)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_dtype", DTYPES)
@pytest.mark.parametrize("cal_sum", [False, True])
@pytest.mark.parametrize(
    "shape", [(129,), (3, 127), (5, 4096), (2, 3, 513), (161, 128)]
)
def test_int8_sum(backend, dtype, scale_dtype, cal_sum, shape):
    torch.manual_seed(170023)
    x = make(shape, dtype)
    if x.numel() > shape[-1]:
        x.reshape(-1, shape[-1])[0].zero_()
    expected = int8_reference(x, scale_dtype, cal_sum)
    actual = load("per_token_quant_int8_sum", backend)(x, scale_dtype, cal_sum)
    assert len(actual) == len(expected)
    delta = (actual[0].to(torch.int16) - expected[0].to(torch.int16)).abs()
    assert delta.max() <= 1
    assert (delta != 0).float().mean() < 0.001
    for a, e in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cal_sum", [False, True])
@pytest.mark.parametrize(
    "shape",
    [(1, 512), (37, 384), (257, 128), (257, 768), (1537, 512), (1537, 1024)],
)
@pytest.mark.parametrize("backend", ["generic", "ascend"])
def test_int8_sum_launch_tiles(dtype, cal_sum, shape, backend):
    torch.manual_seed(170024)
    x = make(shape, dtype, strided=True)
    expected = int8_reference(x, torch.float32, cal_sum)
    actual = load("per_token_quant_int8_sum", backend)(
        x, torch.float32, cal_sum
    )
    delta = (actual[0].to(torch.int16) - expected[0].to(torch.int16)).abs()
    assert delta.max() <= 1
    assert (delta != 0).float().mean() < 0.001
    for a, e in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_int8_sum_ties(backend, dtype):
    half = torch.arange(-63, 64, device="cuda", dtype=torch.float32) + 0.5
    x = (
        torch.cat((half, torch.tensor([127.0], device="cuda")))
        .to(dtype)
        .reshape(1, -1)
    )
    expected = int8_reference(x, torch.float32, True)
    actual = load("per_token_quant_int8_sum", backend)(x, torch.float32, True)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


def fp4_reference(x):
    groups = x.float().reshape(-1, 4, 32)
    sf = (groups.abs().amax(-1) / 6.0).clamp(min=1e-4)
    bits = sf.view(torch.int32)
    exponent = (
        ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    ).clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)
    value = groups / scale[..., None]
    magnitude = value.abs().clamp(max=6.0)
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device
    )
    code = (magnitude[..., None] > thresholds).sum(-1).to(torch.int32)
    code |= ((value < 0) & (code != 0)).to(torch.int32) << 3
    code = code.reshape(-1, 128)
    packed = (
        (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    )
    packed_scale = (
        exponent[:, 0]
        | (exponent[:, 1] << 8)
        | (exponent[:, 2] << 16)
        | (exponent[:, 3] << 24)
    )
    return packed, packed_scale


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "shape", [(1, 128), (5, 128), (2, 3, 128), (161, 128), (4097, 128)]
)
def test_fp4_indexer(backend, dtype, strided, shape):
    torch.manual_seed(172023)
    x = make(shape, dtype, strided)
    x[..., :32].zero_()
    expected = fp4_reference(x)
    actual = load("quantize_fp4_indexer", backend)(x)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES[:2])
def test_fp4_indexer_all_bits(backend, dtype):
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
        .reshape(-1, 128)
    )
    expected = fp4_reference(x)
    actual = load("quantize_fp4_indexer", backend)(x)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp4_indexer_thresholds(backend):
    values = torch.tensor(
        [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda"
    )
    values = torch.cat((values, -values))
    below = torch.nextafter(values, torch.full_like(values, -float("inf")))
    above = torch.nextafter(values, torch.full_like(values, float("inf")))
    all_values = torch.cat((values, below, above))
    x = all_values.repeat(128)[: 128 * 32].reshape(-1, 128)
    for factor in [2.0**-12, 1.0, 4096.0]:
        value = x * factor
        for a, e in zip(
            load("quantize_fp4_indexer", backend)(value), fp4_reference(value)
        ):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("exponent", [-13, -1, 0, 10, 64, 124, 125, 126])
def test_fp4_indexer_scaled_boundaries(backend, exponent):
    scale = 2.0**exponent
    thresholds = (
        torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
        * scale
    )
    values = torch.cat(
        (
            thresholds,
            torch.nextafter(
                thresholds, torch.full_like(thresholds, -float("inf"))
            ),
            torch.nextafter(
                thresholds, torch.full_like(thresholds, float("inf"))
            ),
        )
    )
    x = torch.zeros((8, 128), device="cuda")
    x.view(-1, 32)[:, :21] = values
    x.view(-1, 32)[:, 31] = min(6 * scale, torch.finfo(torch.float32).max)
    x[::2] *= -1
    for actual, expected in zip(
        load("quantize_fp4_indexer", backend)(x), fp4_reference(x)
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "special", [float("nan"), float("inf"), -float("inf")]
)
def test_fp4_indexer_nonfinite_groups(backend, special):
    torch.manual_seed(172026)
    x = torch.randn((16, 128), device="cuda") * 1e37
    x[:, 0] = special
    for actual, expected in zip(
        load("quantize_fp4_indexer", backend)(x), fp4_reference(x)
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

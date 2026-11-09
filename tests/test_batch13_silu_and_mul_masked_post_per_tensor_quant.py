# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import re
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = "silu_and_mul_masked_post_per_tensor_quant"
# Each backend file that exists is loaded from its path, so a chip's own copy
# is checked on this card too. Kunlunxin's compiler takes hints and launch
# options no other Triton accepts; they change how the kernel is scheduled,
# not what it computes, so its copy is checked here with them taken out.
BACKENDS = [
    backend
    for backend in [
        "generic",
        "ascend",
        "kunlunxin",
        "metax",
        "iluvatar",
        "hygon",
        "nvidia",
        "amd",
    ]
    if backend == "generic"
    or (
        ROOT
        / f"src/flaggems_sglang/runtime/backend/_{backend}/ops/{OPERATOR}.py"
    ).exists()
]
VENDOR_ONLY = re.compile(
    r',\s*offset_state_policy="[a-z_]+"'
    r"|^\s*(isCloseMemoryAsync|isCloseUnrollControl|isCloseVectorization"
    r"|isCloseDtypeConvert|groups_per_cluster|vrf_budget|buffer_size_limit)"
    r"=[^\n]*,\n",
    re.M,
)
FP8 = torch.float8_e4m3fn
# The task's comparison: a boundary-tolerant mismatch fraction over each
# expert's valid rows, at atol=rtol=2e-2 with the fraction under 3e-2. The
# kernels encode E4M3FN exactly as torch's cast does, so far fewer than the
# task's 3e-2 are expected; the gate sigmoid's last fp32 ulp can still flip
# a rounding boundary, which is what the slack covers.
ATOL = RTOL = 2e-2
MAX_MISMATCH = 1e-3


def load(backend="generic"):
    folder = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    path = ROOT / f"src/flaggems_sglang/{folder}/{OPERATOR}.py"
    if backend == "kunlunxin":
        # Triton reads a kernel's source back from its file, so the copy
        # without the vendor-only arguments has to be a file as well.
        stripped = Path(tempfile.mkdtemp()) / path.name
        stripped.write_text(VENDOR_ONLY.sub("", path.read_text()))
        path = stripped
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, OPERATOR)


def reference(input, masked_m, scale):
    """The task's reference implementation, copied from its statement."""
    E, T, H2 = input.shape
    H = H2 // 2
    fp8_max = torch.finfo(FP8).max
    fp8_min = -fp8_max
    s = scale.reshape(()).float()

    out = torch.zeros(E, T, H, dtype=FP8, device=input.device)
    for e in range(E):
        n = int(masked_m[e].item())
        if n <= 0:
            continue
        gate = input[e, :n, :H].float()
        up = input[e, :n, H:].float()
        val = gate * torch.sigmoid(gate) * up
        out[e, :n] = (val / s).clamp(fp8_min, fp8_max).to(FP8)
    return out


def make(experts, tokens, half, dtype=torch.bfloat16, seed=0, mask="random"):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input = torch.randn(
        (experts, tokens, 2 * half),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(dtype)
    if mask == "random":
        masked_m = torch.randint(
            0, tokens + 1, (experts,), device="cuda", generator=generator
        )
    elif mask == "full":
        masked_m = torch.full((experts,), tokens, device="cuda")
    elif mask == "empty":
        masked_m = torch.zeros(experts, device="cuda", dtype=torch.int32)
    elif mask == "over":
        masked_m = torch.full((experts,), tokens + 3, device="cuda")
    scale = (
        torch.rand((1,), device="cuda", generator=generator) * 0.02 + 0.001
    )
    return input, masked_m.to(torch.int32), scale


def assert_close(backend, actual, expected, masked_m):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_contiguous()
    experts, tokens, half = actual.shape
    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    for expert in range(experts):
        n = min(max(int(masked_m[expert].item()), 0), tokens)
        if n:
            mismatch = (
                (~torch.isclose(a[expert, :n], e[expert, :n], atol=ATOL, rtol=RTOL))
                .float()
                .mean()
                .item()
            )
            assert mismatch < MAX_MISMATCH, (backend, expert, mismatch)
        if n < tokens:
            assert (actual[expert, n:].view(torch.uint8) == 0).all(), (
                backend,
                expert,
            )


def check(backend, *args, **kwargs):
    input, masked_m, scale = make(*args, **kwargs)
    before = input.clone()
    expected = reference(input, masked_m, scale)
    actual = load(backend)(input, masked_m, scale)
    assert actual.data_ptr() != input.data_ptr()
    assert_close(backend, actual, expected, masked_m)
    # The reference returns a new tensor, so the input comes back untouched.
    assert torch.equal(input, before)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "experts, tokens, half",
    [
        (1, 1, 8),
        (2, 3, 16),
        (8, 64, 512),
        (4, 300, 1024),
        (3, 1000, 1408),
        (8, 1, 4096),
        (2, 4096, 2048),
    ],
)
@pytest.mark.parametrize("mask", ["random", "full", "empty", "over"])
def test_silu_and_mul_masked_post_per_tensor_shapes(
    backend, experts, tokens, half, mask
):
    check(backend, experts, tokens, half, mask=mask)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("mask", ["random", "full"])
def test_silu_and_mul_masked_post_per_tensor_dtypes(backend, dtype, mask):
    check(backend, 5, 257, 768, dtype=dtype, mask=mask)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("half", [1, 2, 3, 5, 6, 7, 48, 96])
def test_silu_and_mul_masked_post_per_tensor_odd_halves(backend, half):
    # A half row that is no multiple of four takes Kunlunxin's byte path.
    check(backend, 3, 129, half)


@pytest.mark.parametrize("backend", BACKENDS)
def test_silu_and_mul_masked_post_per_tensor_strided_input(backend):
    # A last-dim stride of two, as a slice of a wider projection.
    input, masked_m, scale = make(4, 100, 256)
    wide = torch.randn(4, 100, 1024, device="cuda").to(input.dtype)
    wide[..., :512] = input
    expected = reference(input, masked_m, scale)
    actual = load(backend)(wide[..., :512], masked_m, scale)
    assert_close(backend, actual, expected, masked_m)


@pytest.mark.parametrize("backend", BACKENDS)
def test_silu_and_mul_masked_post_per_tensor_transposed(backend):
    # Leading dims that do not collapse: rows are not adjacent.
    base = torch.randn(300, 4, 512, device="cuda").to(torch.bfloat16)
    input = base.transpose(0, 1)
    masked_m = torch.randint(0, 301, (4,), device="cuda", dtype=torch.int32)
    scale = torch.rand((1,), device="cuda") * 0.02 + 0.001
    expected = reference(input, masked_m, scale)
    actual = load(backend)(input, masked_m, scale)
    assert_close(backend, actual, expected, masked_m)


@pytest.mark.parametrize("backend", BACKENDS)
def test_silu_and_mul_masked_post_per_tensor_expanded_mask(backend):
    # A stride of zero from expand must compute, not fail.
    input, _, scale = make(6, 77, 128)
    masked_m = torch.full((1,), 50, device="cuda", dtype=torch.int32)
    masked_m = masked_m.expand(6)
    expected = reference(input, masked_m, scale)
    actual = load(backend)(input, masked_m, scale)
    assert_close(backend, actual, expected, masked_m)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("experts, tokens", [(0, 8), (4, 0)])
def test_silu_and_mul_masked_post_per_tensor_empty(backend, experts, tokens):
    input, masked_m, scale = make(max(experts, 1), max(tokens, 1), 64)
    input = input[:experts, :tokens]
    masked_m = masked_m[:experts]
    out = load(backend)(input, masked_m, scale)
    assert out.shape == (experts, tokens, 64)
    assert out.dtype == FP8


@pytest.mark.parametrize("backend", BACKENDS)
def test_silu_and_mul_masked_post_per_tensor_int64_mask(backend):
    input, masked_m, scale = make(3, 90, 320)
    masked_m = masked_m.to(torch.int64)
    expected = reference(input, masked_m, scale)
    actual = load(backend)(input, masked_m, scale)
    assert_close(backend, actual, expected, masked_m)


@pytest.mark.parametrize("backend", BACKENDS)
def test_silu_and_mul_masked_post_per_tensor_saturation(backend):
    # Values past the E4M3FN range saturate at 448, infinities included, and
    # a NaN stays a NaN.
    experts, tokens, half = 2, 16, 64
    input = torch.zeros(experts, tokens, 2 * half, device="cuda")
    input[0, 0, 0] = 100.0  # 100*sigmoid(100)*100 = 1e4, past 448 after /s
    input[0, 1, 0] = float("inf")
    input[0, 1, half] = 1.0
    input[0, 2, 0] = float("nan")
    input[0, 3, 0] = -1e30
    input[0, 3, half] = 1e30
    input = input.to(torch.bfloat16)
    masked_m = torch.full((experts,), tokens, device="cuda", dtype=torch.int32)
    scale = torch.tensor([0.01], device="cuda")
    expected = reference(input, masked_m, scale)
    actual = load(backend)(input, masked_m, scale)
    assert_close(backend, actual, expected, masked_m)


@pytest.mark.parametrize("backend", BACKENDS)
def test_silu_and_mul_masked_post_per_tensor_scale_view(backend):
    # The scale is read as scale[0] however its storage is laid out.
    input, masked_m, _ = make(2, 40, 160)
    storage = torch.tensor([0.013, 99.0, 0.0], device="cuda")
    scale = storage[:1]
    expected = reference(input, masked_m, scale)
    actual = load(backend)(input, masked_m, scale)
    assert_close(backend, actual, expected, masked_m)

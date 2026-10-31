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
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = "per_token_group_quant_mla_masked_fp8"
FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)


def load(backend="generic"):
    folder = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    path = ROOT / f"src/flaggems_sglang/{folder}/{OPERATOR}.py"
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def reference(x, group_size, eps):
    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256  # 256 is the gemm kernel's max block_m
    num_tiles_k = k // group_size

    x32 = x.to(torch.float32).reshape(b, m, num_tiles_k, group_size)
    absmax = x32.abs().amax(dim=-1).clamp(min=eps)  # [b, m, num_tiles_k]
    scale = absmax / FP8_MAX
    q = (x32 / scale[..., None]).clamp(-FP8_MAX, FP8_MAX)

    # Rows [m, aligned_m) are padding the kernel never writes; zero them here
    # so the reference is well defined, and let the check compare only [:m].
    x_q = torch.zeros((b, aligned_m, k), dtype=FP8, device=x.device)
    x_q[:, :m] = q.reshape(b, m, k).to(FP8)
    x_s = torch.zeros(
        (b, aligned_m, num_tiles_k), dtype=torch.float32, device=x.device
    )
    x_s[:, :m] = scale
    masked_m = torch.full((b,), m, dtype=torch.int32, device=x.device)

    return x_q, x_s, masked_m, m, aligned_m


def assert_matches(actual, expected):
    x_q, x_s, masked_m, m, aligned_m = actual
    want_q, want_s, want_m, want_rows, want_aligned = expected
    assert m == want_rows and aligned_m == want_aligned
    assert x_q.shape == want_q.shape and x_q.dtype == FP8
    assert x_s.shape == want_s.shape and x_s.dtype == torch.float32
    # The masked GEMM reads the scales transposed, so they stay that way.
    tiles = x_s.shape[2]
    assert x_s.stride() == (tiles * aligned_m, 1, aligned_m)
    torch.testing.assert_close(masked_m, want_m, atol=0, rtol=0)
    # Scales are exact: the reference divides the same absmax by the same
    # constant.
    torch.testing.assert_close(x_s[:, :m], want_s[:, :m], atol=0, rtol=0)
    got = x_q[:, :m].float()
    good = want_q[:, :m].float()
    close = torch.isclose(got, good, atol=2e-2, rtol=2e-2) | (
        torch.isnan(got) & torch.isnan(good)
    )
    mismatched = int((~close).sum().item())
    assert (
        mismatched / close.numel() < 3e-2
    ), f"mismatch fraction {mismatched / close.numel():.4f}"


def make(b, m, k, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return torch.randn((b, m, k), device="cuda", dtype=dtype)


CASES = [
    (1, 1, 128),
    (1, 7, 256),
    (2, 64, 512),
    (3, 97, 512),
    (4, 128, 1536),
    (2, 256, 1024),
    (8, 257, 2048),
    (2, 1024, 4096),
]

BACKENDS = ["generic", "kunlunxin", "ascend"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b,m,k", CASES)
def test_mla_masked_shapes(backend, b, m, k):
    x = make(b, m, k, seed=b * 131 + k)
    args = (x, 128, 1e-12)
    assert_matches(load(backend)(*args), reference(*args))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("group_size", [64, 128, 256])
def test_mla_masked_group_sizes(backend, group_size):
    x = make(2, 33, 1024, seed=group_size)
    args = (x, group_size, 1e-12)
    assert_matches(load(backend)(*args), reference(*args))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mla_masked_dtypes(backend, dtype):
    x = make(2, 48, 512, dtype=dtype, seed=7)
    args = (x, 128, 1e-12)
    assert_matches(load(backend)(*args), reference(*args))


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_masked_eps_floor(backend):
    # A row of zeros takes the eps floor, so its scale must come from eps
    # rather than from a division by zero.
    x = make(2, 40, 512, seed=3)
    x[0, 5] = 0.0
    x[1, 39] = 0.0
    args = (x, 128, 1e-10)
    assert_matches(load(backend)(*args), reference(*args))


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_masked_extreme_magnitudes(backend):
    # Rows that are tiny or huge keep the scale in range and the codes
    # inside E4M3FN.
    x = make(2, 24, 512, seed=9)
    x[0, 0] *= 2.0**-20
    x[0, 1] *= 2.0**15
    x[1, 23] *= 2.0**20
    args = (x, 128, 1e-12)
    assert_matches(load(backend)(*args), reference(*args))


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_masked_leaves_input_alone(backend):
    x = make(2, 64, 1024, seed=11)
    before = x.clone()
    load(backend)(x, 128, 1e-12)
    torch.testing.assert_close(x, before, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_masked_padding_rows_untouched(backend):
    # The alignment padding is undefined, but the kernel must stay inside
    # the buffer it was given, so a second run has to repeat the first.
    x = make(2, 100, 512, seed=13)
    first = load(backend)(x, 128, 1e-12)
    second = load(backend)(x, 128, 1e-12)
    torch.testing.assert_close(
        first[0][:, :100].view(torch.uint8),
        second[0][:, :100].view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        first[1][:, :100], second[1][:, :100], atol=0, rtol=0
    )

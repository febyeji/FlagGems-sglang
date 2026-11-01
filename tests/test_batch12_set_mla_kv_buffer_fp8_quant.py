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
OPERATOR = "set_mla_kv_buffer_fp8_quant"
FP8 = torch.float8_e4m3fn
BACKENDS = ["generic", "kunlunxin", "ascend"]


def load(backend="generic"):
    folder = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    path = ROOT / f"src/flaggems_sglang/{folder}/{OPERATOR}.py"
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope, fp8_dtype):
    out = kv_buffer.clone()
    # The buffer is reinterpreted as fp8: the cast happens at the store.
    view = out.view(fp8_dtype)
    row = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
    view[loc.to(torch.int64)] = row.to(fp8_dtype)
    return out


def make(
    n_loc,
    slots=65536,
    nope_dim=512,
    rope_dim=64,
    dtype=torch.bfloat16,
    fp8_dtype=FP8,
    seed=0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # The buffer holds the source dtype; its fp8 view is twice as wide.
    kv_buffer = torch.zeros(
        slots, (nope_dim + rope_dim) // 2, dtype=dtype, device="cuda"
    )
    loc = torch.randperm(slots, generator=g)[:n_loc].to(torch.int64).cuda()
    return dict(
        kv_buffer=kv_buffer,
        loc=loc,
        cache_k_nope=torch.randn(n_loc, nope_dim, generator=g)
        .to(dtype)
        .cuda(),
        cache_k_rope=torch.randn(n_loc, rope_dim, generator=g)
        .to(dtype)
        .cuda(),
        fp8_dtype=fp8_dtype,
    )


def assert_matches(solution, case):
    want = reference(**case)
    # The reference clones; the operator writes in place, as SGLang does,
    # so it starts from the buffer the reference was given.
    got = solution(**case)
    assert got.dtype == want.dtype and got.shape == want.shape
    # Exact on the raw fp8 bytes, which is what the task asks for.
    torch.testing.assert_close(
        got.view(torch.uint8), want.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_loc", [1, 37, 512, 4096])
def test_mla_kv_fp8_cases(backend, n_loc):
    assert_matches(load(backend), make(n_loc, seed=n_loc))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mla_kv_fp8_every_source_value(backend, dtype):
    # Both source dtypes hold 65536 values, so one case covers every one of
    # them and the encoding is checked exhaustively rather than on a sample.
    case = make(128, dtype=dtype, seed=4)
    patterns = torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16)
    case["cache_k_nope"] = (
        patterns.view(dtype).reshape(128, 512).clone().cuda()
    )
    assert_matches(load(backend), case)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fp8_dtype", [FP8, torch.float8_e5m2])
def test_mla_kv_fp8_formats(backend, fp8_dtype):
    # E4M3FN saturates to 448 and E5M2 overflows to infinity; both encoders
    # see every source value here.
    case = make(128, fp8_dtype=fp8_dtype, seed=5)
    patterns = torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16)
    case["cache_k_nope"] = (
        patterns.view(torch.bfloat16).reshape(128, 512).clone().cuda()
    )
    assert_matches(load(backend), case)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_kv_fp8_extreme_magnitudes(backend):
    # Values past the format's range, down in its subnormals, and at the
    # rounding boundary between the two.
    case = make(64, seed=6)
    nope = case["cache_k_nope"]
    nope[0, :8] = torch.tensor(
        [448.0, 456.0, 464.0, 480.0, 1e4, float("inf"), 0.0, -0.0],
        dtype=nope.dtype,
        device=nope.device,
    )
    nope[1, :6] = torch.tensor(
        [2.0**-9, 2.0**-10, 2.0**-11, 1.5 * 2.0**-10, -(2.0**-9), 2.0**-6],
        dtype=nope.dtype,
        device=nope.device,
    )
    nope[2, :2] = torch.tensor(
        [float("nan"), -float("nan")], dtype=nope.dtype, device=nope.device
    )
    assert_matches(load(backend), case)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "nope_dim,rope_dim", [(512, 64), (128, 64), (64, 64), (256, 128)]
)
def test_mla_kv_fp8_widths(backend, nope_dim, rope_dim):
    # The column block that straddles the nope/rope seam is the one worth
    # getting right, so the widths move it around.
    case = make(96, nope_dim=nope_dim, rope_dim=rope_dim, seed=nope_dim)
    assert_matches(load(backend), case)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_kv_fp8_leaves_other_slots_alone(backend):
    # Only the rows named by loc may change; every other slot keeps the
    # bytes it had.
    case = make(64, slots=4096, seed=7)
    case["kv_buffer"].normal_()
    before = case["kv_buffer"].clone()
    assert_matches(load(backend), case)
    untouched = torch.ones(4096, dtype=torch.bool, device="cuda")
    untouched[case["loc"]] = False
    torch.testing.assert_close(
        case["kv_buffer"][untouched].view(torch.uint8),
        before[untouched].view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_mla_kv_fp8_leaves_sources_alone(backend):
    case = make(128, seed=8)
    nope = case["cache_k_nope"].clone()
    rope = case["cache_k_rope"].clone()
    loc = case["loc"].clone()
    load(backend)(**case)
    for actual, expected in (
        (case["cache_k_nope"], nope),
        (case["cache_k_rope"], rope),
        (case["loc"], loc),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

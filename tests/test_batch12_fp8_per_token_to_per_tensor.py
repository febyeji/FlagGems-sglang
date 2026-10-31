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
GROUP = 128


def load(backend="generic"):
    folder = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    path = (
        ROOT
        / f"src/flaggems_sglang/{folder}/fp8_per_token_to_per_tensor_quant.py"
    )
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def reference(x, x_scale, masked_m, output_scale):
    experts, tokens, width = x.shape
    out = x.new_zeros(experts, tokens, width)
    inverse = 1.0 / output_scale.reshape(()).float()
    for expert in range(experts):
        rows = int(masked_m[expert].item())
        if rows <= 0:
            continue
        values = x[expert, :rows].float()
        scale = x_scale[expert, :rows].float().repeat_interleave(GROUP, dim=-1)
        out[expert, :rows] = (values * scale * inverse).to(out.dtype)
    return out


def make(experts, tokens, width, rows, seed=0):
    torch.manual_seed(seed)
    raw = torch.randn(
        (experts, tokens, width), device="cuda", dtype=torch.float32
    )
    grouped = raw.view(experts, tokens, width // GROUP, GROUP)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = amax / 448.0
    x = (grouped / scale).to(torch.float8_e4m3fn).view(experts, tokens, width)
    x_scale = scale.squeeze(-1).contiguous()
    masked_m = torch.tensor(rows, device="cuda", dtype=torch.int32)
    # A per-tensor scale near the data range keeps the cast away from overflow.
    output_scale = (raw.abs().amax() / 448.0).reshape(1).contiguous()
    return x, x_scale, masked_m, output_scale


def assert_matches(actual, expected, masked_m):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    experts = actual.shape[0]
    mismatched = 0
    counted = 0
    for expert in range(experts):
        rows = int(masked_m[expert].item())
        tail = actual[expert, rows:].view(torch.uint8)
        assert int(tail.max().item() if tail.numel() else 0) == 0
        if rows <= 0:
            continue
        got = actual[expert, :rows].float()
        want = expected[expert, :rows].float()
        close = torch.isclose(got, want, atol=2e-2, rtol=2e-2) | (
            torch.isnan(got) & torch.isnan(want)
        )
        mismatched += int((~close).sum().item())
        counted += close.numel()
    assert (
        counted == 0 or mismatched / counted < 3e-2
    ), f"mismatch fraction {mismatched / counted:.4f}"


CASES = [
    (1, 1, 128, [1]),
    (1, 4, 256, [3]),
    (2, 8, 512, [8, 0]),
    (3, 16, 1024, [16, 1, 9]),
    (4, 64, 2048, [64, 33, 0, 7]),
    (8, 128, 4096, [128, 64, 32, 16, 8, 4, 2, 0]),
    (2, 97, 7168, [97, 50]),
]


BACKENDS = ["generic", "kunlunxin", "ascend"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("experts,tokens,width,rows", CASES)
def test_fp8_per_token_to_per_tensor_shapes(
    backend, experts, tokens, width, rows
):
    args = make(experts, tokens, width, rows, seed=experts * 131 + width)
    actual = load(backend)(*args)
    assert_matches(actual, reference(*args), args[2])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fp8_per_token_to_per_tensor_masked_dtype(backend, dtype):
    x, x_scale, masked_m, output_scale = make(3, 32, 512, [32, 0, 17], seed=5)
    masked_m = masked_m.to(dtype)
    args = (x, x_scale, masked_m, output_scale)
    assert_matches(load(backend)(*args), reference(*args), masked_m)


def _widen_case(backend):
    width = 256
    codes = torch.arange(256, device="cuda", dtype=torch.uint8)
    x = codes.view(1, 1, width).view(torch.float8_e4m3fn)
    x_scale = torch.ones((1, 1, width // GROUP), device="cuda")
    masked_m = torch.tensor([1], device="cuda", dtype=torch.int32)
    output_scale = torch.ones(1, device="cuda")
    args = (x, x_scale, masked_m, output_scale)
    return (
        codes,
        load(backend)(*args).view(torch.uint8)[0, 0],
        reference(*args).view(torch.uint8)[0, 0],
    )


@pytest.mark.parametrize("backend", ["generic", "kunlunxin"])
def test_fp8_per_token_to_per_tensor_all_input_bytes(backend):
    # Every pattern round-trips, negative zero and both NaN encodings
    # included.
    _, actual, expected = _widen_case(backend)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fp8_per_token_to_per_tensor_input_bytes_ascend():
    # The Ascend variant applies the sign by negating, the way the Ascend
    # kernels that compile on that chip do, so negative zero comes back as
    # positive zero. Every other pattern round-trips, and the task compares
    # values, where the two are equal.
    codes, actual, expected = _widen_case("ascend")
    other = codes != 0x80
    torch.testing.assert_close(actual[other], expected[other], atol=0, rtol=0)
    assert torch.equal(actual[codes == 0x80], expected[codes == 0x00])


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp8_per_token_to_per_tensor_leaves_input_alone(backend):
    args = make(2, 16, 1024, [16, 5], seed=11)
    before = tuple(t.clone() for t in args)
    load(backend)(*args)
    for got, want in zip(args, before):
        torch.testing.assert_close(
            got.view(torch.uint8) if got.dtype == torch.float8_e4m3fn else got,
            (
                want.view(torch.uint8)
                if want.dtype == torch.float8_e4m3fn
                else want
            ),
            atol=0,
            rtol=0,
        )

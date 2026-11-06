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
OPERATOR = "bias_gelu_tanh"
VENDORS = [
    "ascend",
    "kunlunxin",
    "metax",
    "iluvatar",
    "hygon",
    "nvidia",
    "amd",
]
# The task's own tolerance per dtype. The sum is rounded once to the
# input dtype, so what the kernel is allowed to differ by is one
# rounding step of that dtype.
TOLERANCE = [
    (torch.bfloat16, dict(atol=1.5e-2, rtol=1.5e-2)),
    (torch.float16, dict(atol=1e-2, rtol=1e-2)),
]


def path_of(backend):
    folder = "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    return ROOT / f"src/flaggems_sglang/{folder}/{OPERATOR}.py"


# Every chip runs the file named after its vendor and falls back to the
# generic one, so a backend with no file of its own is nothing to test.
BACKENDS = ["generic"] + [v for v in VENDORS if path_of(v).exists()]


def load(backend="generic"):
    path = path_of(backend)
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def tolerance(dtype):
    return next(tol for key, tol in TOLERANCE if key == dtype)


def reference(x, bias):
    """The task's reference implementation, copied from its statement."""
    biased = (x.float() + bias.float()).to(x.dtype)
    return torch.nn.functional.gelu(biased, approximate="tanh")


def make(shape, dtype=torch.bfloat16, seed=0, scale=1.0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*size):
        return (
            torch.randn(*size, generator=g, device=device, dtype=torch.float32)
            * scale
        ).to(dtype)

    return randn(*shape), randn(shape[-1])


def assert_matches(solution, case):
    x, bias = case
    expected = reference(x, bias)
    before = (x.clone(), bias.clone())
    actual = solution(x, bias)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    # A new tensor, not a view of either input and not one of them.
    assert actual.data_ptr() not in (x.data_ptr(), bias.data_ptr())
    torch.testing.assert_close(
        actual.to(torch.float32),
        expected.to(torch.float32),
        **tolerance(x.dtype),
    )
    for seen, kept in zip((x, bias), before):
        torch.testing.assert_close(seen, kept, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
@pytest.mark.parametrize(
    "shape",
    [(1, 8), (1, 4096), (37, 1024), (128, 11008), (4096, 4096), (8192, 2048)],
)
def test_bias_gelu_tanh_cases(backend, dtype, _tol, shape):
    assert_matches(load(backend), make(shape, dtype=dtype, seed=shape[0]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("h", [8, 16, 24, 40, 56, 72, 88, 120, 1368, 11008])
def test_bias_gelu_tanh_widths(backend, h):
    # H is a multiple of 8 and nothing more, so the row width is often
    # neither a power of two nor a multiple of the block.
    assert_matches(load(backend), make((131, h), seed=h))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "shape", [(3, 5, 64), (2, 7, 11, 128), (6, 1, 3, 2, 32)]
)
def test_bias_gelu_tanh_leading_dims(backend, shape):
    # The signature allows any number of leading dimensions; the bias
    # covers the last one alone.
    assert_matches(load(backend), make(shape, seed=len(shape)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_bias_gelu_tanh_rounds_before_gelu(backend, dtype, _tol):
    # The sum is rounded to the input dtype before GELU. That rounding
    # moves z by at most half a step of the dtype, which is inside the
    # task's own tolerance, so `assert_close` cannot see it: the kernel
    # is compared against both readings instead and has to sit on the
    # one the task defines.
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(29)
    x = (
        torch.randn(256, 512, generator=g, device=device, dtype=torch.float32)
        * 2.0
    ).to(dtype)
    # A bias of half a step of x's exponent puts many sums exactly
    # between two representable numbers, where the two readings part.
    bias = (
        torch.randn(512, generator=g, device=device, dtype=torch.float32)
        * torch.finfo(dtype).eps
    ).to(dtype)
    rounded = reference(x, bias)
    unrounded = torch.nn.functional.gelu(
        x.float() + bias.float(), approximate="tanh"
    ).to(dtype)
    # The case has to separate the two in the first place.
    assert not torch.equal(rounded, unrounded)
    out = load(backend)(x, bias).float()
    to_rounded = (out - rounded.float()).abs().sum()
    to_unrounded = (out - unrounded.float()).abs().sum()
    assert to_rounded < to_unrounded
    assert_matches(load(backend), (x, bias))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_bias_gelu_tanh_extreme_values(backend, dtype, _tol):
    # The far negative tail is where a clamped tanh approximation stops
    # agreeing with the reference: GELU is zero there while a tanh cut
    # off at a fixed argument leaves a residue the tolerance rejects.
    x, bias = make((64, 128), dtype=dtype, seed=3)
    limit = 240.0 if dtype == torch.float16 else 3.0e38
    x[0] = limit
    x[1] = -limit
    x[2] = 0.0
    x[3] = torch.finfo(dtype).tiny
    bias[:8] = 0.0
    assert_matches(load(backend), (x, bias))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
@pytest.mark.parametrize("shape", [(3, 24), (1, 4096)])
def test_bias_gelu_tanh_on_another_device(backend, dtype, _tol, shape):
    # Triton launches on the device that is current and rejects a
    # pointer it cannot reach from there, so inputs that live on another
    # device have to make their own device current for the launch and
    # give the current device back afterwards. The two shapes take the
    # masked and the exact path.
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second device to be somewhere else")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    solution = load(backend)
    # A device's context is what a launch runs in, and it becomes the
    # current one when that device is used, so the call below only has
    # somewhere else to be once this one has run.
    solution(*make((8, 8), dtype=dtype, seed=1))
    with torch.cuda.device(other):
        # `make` builds on the current device, which in here is the one
        # the call will not be launched from.
        case = make(shape, dtype=dtype, seed=shape[-1])
    assert case[0].device.index == other
    assert torch.cuda.current_device() == current
    assert_matches(solution, case)
    assert torch.cuda.current_device() == current


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_bias_gelu_tanh_non_contiguous_inputs(backend, dtype, _tol):
    # The kernel walks a flat index, so the launcher packs anything that
    # is not contiguous first. What it packs is a fresh allocation, and
    # so takes the held kernel's aligned path.
    wide, bias_wide = make((64, 2048), dtype=dtype, seed=13)
    x = wide[:, ::2]
    bias = bias_wide[::2]
    assert not x.is_contiguous() and not bias.is_contiguous()
    assert_matches(load(backend), (x, bias))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
@pytest.mark.parametrize("offset", [1, 3, 7])
def test_bias_gelu_tanh_unaligned_inputs(backend, dtype, _tol, offset):
    # A launcher that holds a compiled kernel holds one that Triton
    # specialized on 16-byte aligned pointers. A view into the middle of
    # an allocation is not aligned, and has to reach a kernel that makes
    # no such assumption.
    device = "cuda"
    rows, h = 8, 512
    g = torch.Generator(device=device).manual_seed(11)

    def buffer(n):
        return (
            torch.randn(n + 8, generator=g, device=device, dtype=torch.float32)
        ).to(dtype)

    x = buffer(rows * h)[offset : offset + rows * h].view(rows, h)
    bias = buffer(h)[offset : offset + h]
    assert x.is_contiguous() and bias.is_contiguous()
    # Two bytes an element, so an odd offset lands off the boundary.
    assert x.data_ptr() % 16 and bias.data_ptr() % 16
    assert_matches(load(backend), (x, bias))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_bias_gelu_tanh_on_a_side_stream(backend, dtype, _tol):
    # A launcher that hands Triton a stream has to hand it the one the
    # caller is on, or the kernel races the allocation of its own
    # output and whatever the caller queues next.
    device = "cuda"
    solution = load(backend)
    case = make((512, 1024), dtype=dtype, seed=7)
    expected = reference(*case)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # Work queued ahead of the call on the same stream: a kernel
        # launched elsewhere would not wait for it.
        busy = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        for _ in range(8):
            busy = busy @ busy.T / 4096.0
        actual = solution(*case)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.to(torch.float32),
        expected.to(torch.float32),
        **tolerance(dtype),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_bias_gelu_tanh_second_device_is_current(backend, dtype, _tol):
    # A compiled kernel belongs to the device it was loaded on, so a
    # launcher that holds one has to hold it per device. This runs the
    # whole call on the second device, after the first has been used.
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second device to be somewhere else")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    solution = load(backend)
    solution(*make((128, 256), dtype=dtype, seed=2))
    with torch.cuda.device(other):
        case = make((128, 256), dtype=dtype, seed=2)
        assert case[0].device.index == other
        assert_matches(solution, case)
    # and back, so the two devices' kernels are both exercised
    assert_matches(solution, make((128, 256), dtype=dtype, seed=2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1.0, 8.0, 64.0])
def test_bias_gelu_tanh_magnitudes(backend, scale):
    # Activations entering a bias-GELU are not all near one, and the
    # cubic term inside the tanh grows with the cube of the input.
    assert_matches(load(backend), make((257, 512), seed=17, scale=scale))


@pytest.mark.parametrize("backend", BACKENDS)
def test_bias_gelu_tanh_bias_is_used_per_column(backend):
    # A kernel that broadcast the wrong bias element, or none, would
    # still pass a case whose bias is constant.
    device = "cuda"
    h = 256
    x = torch.zeros(9, h, device=device, dtype=torch.bfloat16)
    bias = (
        torch.arange(h, device=device, dtype=torch.float32) / 64.0 + 0.5
    ).to(torch.bfloat16)
    assert_matches(load(backend), (x, bias))
    # The same output under a rotated bias would mean the column index
    # never reached the load.
    rotated = reference(x, bias.roll(1))
    assert not torch.equal(load(backend)(x, bias), rotated)

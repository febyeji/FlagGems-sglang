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
OPERATOR = "dequantize_k_cache_fast"
# Kunlunxin has no variant of its own: it took the generic file whole,
# and the one warp its own file asked for was what made it fail.
BACKENDS = ["generic", "ascend"]
NOPE_DIM = 512
ROPE_DIM = 64


def load(backend="generic"):
    folder = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    path = ROOT / f"src/flaggems_sglang/{folder}/{OPERATOR}.py"
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def reference(quant_k_cache, group_size=128):
    """The task's reference implementation, copied from its statement."""
    num_tokens, dim_quant = quant_k_cache.shape
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size

    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(
        torch.bfloat16
    )
    output[:, dim_nope:] = input_rope

    for tile_idx in range(num_tiles):
        cur_nope = input_nope_q[
            :, tile_idx * group_size : (tile_idx + 1) * group_size
        ].to(torch.float32)
        cur_scale = input_nope_s[:, tile_idx].unsqueeze(-1)
        output[:, tile_idx * group_size : (tile_idx + 1) * group_size] = (
            cur_nope * cur_scale
        )

    return output


def make(rows, group_size=128, seed=0, bytes_=None, scales=None):
    """Build a packed row: FP8 nope, FP32 scales, BF16 rope, all as bytes."""
    tiles = NOPE_DIM // group_size
    width = NOPE_DIM + tiles * 4 + ROPE_DIM * 2
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randint(0, 256, (rows, width), generator=g, dtype=torch.uint8)
    if bytes_ is not None:
        raw[:, :NOPE_DIM] = bytes_
    if scales is None:
        scales = (
            torch.rand(rows, tiles, generator=g, dtype=torch.float32) * 0.5
            + 0.01
        )
    raw[:, NOPE_DIM : NOPE_DIM + tiles * 4] = scales.view(torch.uint8)
    return raw.cuda().view(torch.float8_e4m3fn)


def assert_matches(solution, case, group_size=128):
    want = reference(case, group_size)
    got = solution(case, group_size)
    assert got.dtype == want.dtype and got.shape == want.shape
    # The arithmetic is the reference's own: widen to FP32, scale, round
    # once to BF16. That leaves no room for a difference, so the check is
    # exact rather than at the task's 1.5e-2.
    torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", [1, 37, 512, 4096])
def test_dequant_k_cache_cases(backend, rows):
    assert_matches(load(backend), make(rows, seed=rows))


@pytest.mark.parametrize("backend", BACKENDS)
def test_dequant_k_cache_every_byte(backend):
    # FP8 holds 256 values, so one case covers every one of them and the
    # decoding is checked exhaustively rather than on a sample.
    rows = 128
    pattern = torch.arange(256, dtype=torch.uint8).repeat(
        rows * NOPE_DIM // 256
    )
    assert_matches(
        load(backend),
        make(rows, seed=1, bytes_=pattern.reshape(rows, NOPE_DIM)),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_dequant_k_cache_extreme_scales(backend):
    # A scale that pushes the product past BF16's range, one that sinks it
    # under BF16's smallest subnormal, a negative one, and zero.
    rows = 64
    tiles = NOPE_DIM // 128
    scales = torch.full((rows, tiles), 1.0, dtype=torch.float32)
    scales[0] = torch.tensor([3.0e38, 1.0e-40, -2.5, 0.0])
    scales[1] = torch.tensor([65504.0, 2.0**-120, -(2.0**120), 1.0e-5])
    pattern = torch.arange(256, dtype=torch.uint8).repeat(
        rows * NOPE_DIM // 256
    )
    assert_matches(
        load(backend),
        make(
            rows,
            seed=2,
            bytes_=pattern.reshape(rows, NOPE_DIM),
            scales=scales,
        ),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("group_size", [512, 256, 128, 64])
def test_dequant_k_cache_group_sizes(backend, group_size):
    # The group size moves the scale and rope regions, and the reference
    # takes it as an argument, so the packing has to follow it.
    assert_matches(
        load(backend),
        make(96, group_size=group_size, seed=group_size),
        group_size=group_size,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_dequant_k_cache_leaves_input_alone(backend):
    case = make(128, seed=3)
    before = case.view(torch.uint8).clone()
    load(backend)(case)
    torch.testing.assert_close(case.view(torch.uint8), before, atol=0, rtol=0)

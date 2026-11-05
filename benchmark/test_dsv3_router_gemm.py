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

"""Benchmark for gemm/dsv3_router_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

SHAPES = [(m, 256, 7168) for m in (1, 2, 4, 8)]
MORE_SHAPES = [(16, 256, 7168), (16, 384, 7168)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_experts, hidden_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    router_weights = torch.randn(
        num_experts,
        hidden_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    yield hidden_states, router_weights, torch.bfloat16


def _case_fn(shape, cur_dtype):
    num_tokens, num_experts, hidden_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "hidden_states": (num_tokens, hidden_dim),
            "router_weights": (num_experts, hidden_dim),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "hidden_dim": hidden_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.dsv3_router_gemm
def test_perf_dsv3_router_gemm():
    bench = OpBenchmark(
        op_name="dsv3_router_gemm",
        torch_op=get_reference("dsv3_router_gemm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_experts, hidden_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()

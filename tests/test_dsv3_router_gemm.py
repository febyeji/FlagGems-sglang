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

"""Correctness test for gemm/dsv3_router_gemm."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

# ---------------------------------------------------------------------------
# Tolerance helper
# ---------------------------------------------------------------------------

# An fp32 result is held to 1e-4, the same tolerance the competition harness
# applies, and that is tight for this reduction: a chain of 7168 fused
# multiply-adds drifts about 4e-4 from an fp32 matmul of the same inputs,
# which is why the kernel splits the reduction until no accumulator spans
# more than 128 elements.
_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype, _DEFAULT_TOLERANCE
        )
    )
    tol.update(overrides)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(
        actual.to(torch.float32), expected.to(torch.float32), **tol
    )


# ---------------------------------------------------------------------------
# Cases: num_tokens in [1, 16], hidden_dim a multiple of 1024,
# num_experts in {256, 384}, out_dtype bfloat16 or float32.
# ---------------------------------------------------------------------------


def _case(
    num_tokens, num_experts=256, hidden_dim=7168, out_dtype=None, seed=0
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, hidden_dim, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    router_weights = torch.randn(
        num_experts, hidden_dim, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    return dict(
        hidden_states=hidden_states,
        router_weights=router_weights,
        out_dtype=out_dtype if out_dtype is not None else torch.bfloat16,
        check=assert_close,
    )


_CASE_FACTORIES = [
    (lambda m=m: _case(m, seed=m)) for m in (1, 2, 3, 4, 7, 8, 13, 16)
]
_CASE_FACTORIES += [
    (lambda m=m: _case(m, out_dtype=torch.float32, seed=100 + m))
    for m in (1, 2, 5, 8, 16)
]
_CASE_FACTORIES += [
    (lambda m=m: _case(m, num_experts=384, seed=200 + m)) for m in (1, 8, 16)
]
_CASE_FACTORIES += [
    (
        lambda m=m: _case(
            m, num_experts=384, out_dtype=torch.float32, seed=300 + m
        )
    )
    for m in (1, 16)
]
_CASE_FACTORIES += [
    (lambda hd=hd: _case(16, hidden_dim=hd, seed=400 + hd))
    for hd in (1024, 2048, 4096)
]
_CASE_FACTORIES += [
    (
        lambda hd=hd: _case(
            3,
            num_experts=384,
            hidden_dim=hd,
            out_dtype=torch.float32,
            seed=500 + hd,
        )
    )
    for hd in (1024, 3072)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))

# 关键约定：bench cases 合并进 accuracy cases


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dsv3_router_gemm
def test_dsv3_router_gemm(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dsv3_router_gemm")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        candidate = get_candidate("dsv3_router_gemm", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("gemm/dsv3_router_gemm ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("gemm/dsv3_router_gemm not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    # Restore check for reuse
    if check is not None:
        case["check"] = check

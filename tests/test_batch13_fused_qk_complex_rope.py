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
OPERATOR = "fused_qk_complex_rope"
BACKENDS = [
    "generic",
    "ascend",
    "kunlunxin",
    "metax",
    "iluvatar",
    "hygon",
    "nvidia",
    "amd",
]
# The task's own tolerance, per dtype. The kernel rounds once to the input
# dtype, so the difference it allows is one rounding step wide.
TOLERANCE = [
    (torch.bfloat16, dict(atol=1.5e-2, rtol=1.5e-2)),
    (torch.float16, dict(atol=1e-2, rtol=1e-2)),
]


def load(backend="generic"):
    folder = (
        "ops" if backend == "generic" else f"runtime/backend/_{backend}/ops"
    )
    path = ROOT / f"src/flaggems_sglang/{folder}/{OPERATOR}.py"
    spec = importlib.util.spec_from_file_location("candidate" + backend, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution


def tolerance(dtype):
    return next(tol for key, tol in TOLERANCE if key == dtype)


def reference(q, k, freqs_cis):
    """The task's reference implementation, copied from its statement."""

    def rotate(x):
        xc = torch.view_as_complex(
            x.to(torch.float32).reshape(*x.shape[:-1], -1, 2)
        )
        out = xc * freqs_cis.unsqueeze(-2)
        return torch.view_as_real(out).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


def make(shape, dtype=torch.bfloat16, strided=False, seed=0, unit=True):
    """Build one case; `strided` makes Q and K views of a packed QKV."""
    *lead, heads, dim = shape
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*size):
        return torch.randn(
            *size, generator=g, device=device, dtype=torch.float32
        ).to(dtype)

    if strided:
        qkv = randn(*lead, 3, heads, dim)
        q, k, _ = torch.unbind(qkv, dim=-3)
    else:
        q, k = randn(*lead, heads, dim), randn(*lead, heads, dim)
    angle = torch.randn(
        *lead, dim // 2, generator=g, device=device, dtype=torch.float32
    )
    # Rope frequencies sit on the unit circle; `unit=False` also scales
    # them, which separates a swapped real/imaginary pair from a correct
    # one by magnitude as well as by sign.
    radius = (
        torch.ones_like(angle)
        if unit
        else torch.rand(angle.shape, generator=g, device=device) * 4 + 0.25
    )
    return q, k, torch.polar(radius, angle)


def assert_matches(solution, case):
    q, k, freqs_cis = case
    expected = reference(q, k, freqs_cis)
    before = (q.clone(), k.clone())
    actual = solution(q, k, freqs_cis)
    assert len(actual) == 2
    for got, want in zip(actual, expected):
        assert got.shape == want.shape and got.dtype == want.dtype
        # The task asks for contiguous outputs whatever the inputs were.
        assert got.is_contiguous()
        torch.testing.assert_close(
            got.to(torch.float32),
            want.to(torch.float32),
            **tolerance(q.dtype),
        )
    for seen, kept in zip((q, k), before):
        torch.testing.assert_close(seen, kept, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
@pytest.mark.parametrize(
    "shape", [(1, 16, 128), (37, 16, 128), (128, 8, 64), (1024, 16, 128)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_fused_qk_complex_rope_cases(backend, dtype, _tol, shape, strided):
    assert_matches(
        load(backend),
        make(shape, dtype=dtype, strided=strided, seed=shape[0]),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [2, 8, 72, 80, 96, 256])
def test_fused_qk_complex_rope_head_dims(backend, dim):
    # Half the head dimension is the pair count, and it is not a power of
    # two for every head size a vision tower uses.
    assert_matches(load(backend), make((64, 5, dim), seed=dim, strided=True))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("heads", [1, 3, 7, 16, 40])
def test_fused_qk_complex_rope_head_counts(backend, heads):
    # One frequency row serves every head, including a head count that no
    # block size divides.
    assert_matches(load(backend), make((33, heads, 64), seed=heads))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(2, 37, 16, 64), (3, 2, 5, 4, 32)])
def test_fused_qk_complex_rope_leading_dims(backend, shape):
    # The signature allows any number of leading dimensions; the
    # frequencies carry the same ones.
    assert_matches(load(backend), make(shape, seed=len(shape)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_qk_complex_rope_scaled_frequencies(backend):
    # A frequency off the unit circle scales the result, so a kernel that
    # swapped the real and imaginary halves cannot pass by symmetry.
    assert_matches(load(backend), make((96, 4, 64), seed=7, unit=False))


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_qk_complex_rope_distinct_q_and_k(backend):
    # Q and K go through one launch; a kernel that wrote one of them
    # twice would pass a case where they hold the same values.
    q, k, freqs_cis = make((64, 8, 128), seed=11)
    q_out, k_out = load(backend)(q, k, freqs_cis)
    assert not torch.equal(q_out, k_out)
    assert_matches(load(backend), (q, k, freqs_cis))


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_qk_complex_rope_extreme_values(backend):
    # Vision activations are small, but the dtype's own range is what the
    # kernel has to carry through fp32 and back.
    q, k, freqs_cis = make((48, 4, 64), seed=13)
    big = torch.tensor(240.0, device=q.device, dtype=q.dtype)
    small = torch.tensor(2.0**-14, device=q.device, dtype=q.dtype)
    q[0] = big
    q[1] = small
    k[2] = -big
    k[3] = 0.0
    assert_matches(load(backend), (q, k, freqs_cis))


def spread(x, filler):
    """A view of `x` whose last-dim stride is two.

    The gaps hold `filler`, so a kernel that walks the row as if it were
    contiguous reads those values in place of its own.
    """
    wide = torch.empty(
        *x.shape[:-1], x.shape[-1] * 2, dtype=x.dtype, device=x.device
    )
    wide[..., ::2] = x
    wide[..., 1::2] = filler
    view = wide[..., ::2]
    assert view.stride(-1) == 2
    return view


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
@pytest.mark.parametrize("strided", [False, True])
def test_fused_qk_complex_rope_strided_frequencies(
    backend, dtype, _tol, strided
):
    # The frequencies may be a slice of a wider table, which leaves
    # their pairs a stride apart. The task's supported inputs do not
    # require the table to be contiguous, and a kernel that takes the
    # real part at twice the pair index reads its neighbour's.
    q, k, freqs_cis = make((5, 3, 64), dtype=dtype, strided=strided, seed=19)
    filler = torch.full_like(freqs_cis, 1j)
    assert_matches(load(backend), (q, k, spread(freqs_cis, filler)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_fused_qk_complex_rope_strided_inputs(backend, dtype, _tol):
    # Q and K may have a last-dim stride of their own, and then the two
    # elements of a pair are not adjacent: there is no 32-bit word to
    # read, and the kernel has to walk the row by its stride.
    q, k, freqs_cis = make((5, 3, 64), dtype=dtype, seed=23)
    filler_q = torch.full_like(q, 7.0)
    filler_k = torch.full_like(k, -3.0)
    assert_matches(
        load(backend), (spread(q, filler_q), spread(k, filler_k), freqs_cis)
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,_tol", TOLERANCE)
def test_fused_qk_complex_rope_odd_storage_offset(backend, dtype, _tol):
    # A contiguous row that starts at an odd element is still a row of
    # pairs, but it cannot be viewed as 32-bit words.
    tokens, heads, dim = 5, 3, 64
    g = torch.Generator(device="cuda").manual_seed(29)
    buffer = torch.randn(
        2 * tokens * heads * dim + 1,
        generator=g,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)
    size = tokens * heads * dim
    q = buffer[1 : 1 + size].view(tokens, heads, dim)
    k = buffer[1 + size : 1 + 2 * size].view(tokens, heads, dim)
    assert q.storage_offset() % 2 == 1 and q.stride(-1) == 1
    _, _, freqs_cis = make((tokens, heads, dim), dtype=dtype, seed=29)
    assert_matches(load(backend), (q, k, freqs_cis))

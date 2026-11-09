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
OPERATOR = "apply_rotary_emb"
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
TOLERANCE = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
}


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


def reference(x, freqs_cis, positions, inverse):
    """The task's reference implementation, copied from its statement."""
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).to(torch.float32)
    rows = (
        torch.arange(x.shape[0], device=x.device)
        if positions is None
        else positions.to(torch.int64)
    )
    f = freqs_real[rows]  # [tokens, rope_dim]
    if x.ndim == 3:
        f = f[:, None, :]
    f_real, f_imag = f[..., 0::2], f[..., 1::2]

    x32 = x.to(torch.float32)
    x_real, x_imag = x32[..., 0::2], x32[..., 1::2]
    if inverse:
        out_real = x_real * f_real + x_imag * f_imag
        out_imag = x_imag * f_real - x_real * f_imag
    else:
        out_real = x_real * f_real - x_imag * f_imag
        out_imag = x_real * f_imag + x_imag * f_real
    return torch.stack([out_real, out_imag], dim=-1).flatten(-2).to(x.dtype)


def make(
    tokens,
    rope_dim,
    dtype=torch.bfloat16,
    heads=None,
    positions=True,
    max_seq=4096,
    row_width=None,
    seed=0,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (tokens, rope_dim) if heads is None else (tokens, heads, rope_dim)
    width = row_width or rope_dim
    storage = torch.randn(
        shape[:-1] + (width,),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    # A width wider than the rope dim leaves x a view of a larger buffer,
    # as when the rope part is sliced out of a projection.
    x = storage.to(dtype)[..., :rope_dim]
    rows = max_seq if positions else tokens
    angle = torch.rand(
        rows, rope_dim // 2, device="cuda", generator=generator
    ) * (2 * torch.pi)
    freqs_cis = torch.polar(torch.ones_like(angle), angle)
    pos = (
        torch.randint(
            0, max_seq, (tokens,), device="cuda", generator=generator
        )
        if positions
        else None
    )
    return x, freqs_cis, pos


def check(backend, *args, inverse=False, **kwargs):
    x, freqs_cis, pos = make(*args, **kwargs)
    before = x.clone()
    expected = reference(x, freqs_cis, pos, inverse)
    actual = load(backend)(x, freqs_cis, pos, inverse)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.data_ptr() != x.data_ptr()
    torch.testing.assert_close(actual, expected, **TOLERANCE[x.dtype])
    # The reference returns a new tensor, so the input comes back untouched.
    assert torch.equal(x, before)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("tokens", [1, 3, 64, 1000, 4096])
@pytest.mark.parametrize("positions", [True, False])
@pytest.mark.parametrize("inverse", [False, True])
def test_apply_rotary_emb_2d(backend, dtype, tokens, positions, inverse):
    # DeepSeek's per-token rope part: a 64-wide row of 32 complex pairs.
    check(
        backend, tokens, 64, dtype, positions=positions, inverse=inverse
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens, heads", [(1, 16), (7, 1), (128, 8), (333, 128)])
@pytest.mark.parametrize("positions", [True, False])
@pytest.mark.parametrize("inverse", [False, True])
def test_apply_rotary_emb_3d(
    backend, dtype, tokens, heads, positions, inverse
):
    # The attention-output form: every head of a token takes the same row.
    check(
        backend,
        tokens,
        64,
        dtype,
        heads=heads,
        positions=positions,
        inverse=inverse,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rope_dim", [2, 6, 16, 32, 96, 128, 192, 256])
@pytest.mark.parametrize("positions", [True, False])
def test_apply_rotary_emb_widths(backend, rope_dim, positions):
    check(backend, 257, rope_dim, positions=positions)
    check(backend, 65, rope_dim, heads=3, positions=positions)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("heads", [None, 4])
def test_apply_rotary_emb_row_view(backend, heads):
    # A rope part sliced out of a wider buffer: rows are not adjacent.
    check(backend, 300, 64, heads=heads, row_width=192)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("inverse", [False, True])
def test_apply_rotary_emb_int32_positions(backend, inverse):
    x, freqs_cis, pos = make(129, 64)
    pos = pos.to(torch.int32)
    torch.testing.assert_close(
        load(backend)(x, freqs_cis, pos, inverse),
        reference(x, freqs_cis, pos, inverse),
        **TOLERANCE[x.dtype],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_rotary_emb_freqs_view(backend):
    # A table sliced from a longer one keeps its row stride.
    x, _, pos = make(96, 64)
    angle = torch.rand(4096, 64, device="cuda") * (2 * torch.pi)
    sliced = torch.polar(torch.ones_like(angle), angle)[:, :32]
    torch.testing.assert_close(
        load(backend)(x, sliced, pos, False),
        reference(x, sliced, pos, False),
        **TOLERANCE[x.dtype],
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("heads", [None, 8])
def test_apply_rotary_emb_empty(backend, heads):
    x, freqs_cis, pos = make(0, 64, heads=heads)
    out = load(backend)(x, freqs_cis, pos, False)
    assert out.shape == x.shape and out.dtype == x.dtype


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tokens", [1, 7, 8, 74, 65536])
def test_apply_rotary_emb_counts(backend, tokens):
    # Counts that do and do not divide into whole programs.
    check(backend, tokens, 64, positions=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_rotary_emb_repeated_positions(backend):
    # Several tokens on one frequency row, as in a batched decode step.
    x, freqs_cis, _ = make(512, 64)
    pos = torch.full((512,), 17, device="cuda", dtype=torch.int64)
    torch.testing.assert_close(
        load(backend)(x, freqs_cis, pos, False),
        reference(x, freqs_cis, pos, False),
        **TOLERANCE[x.dtype],
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("heads", [None, 8])
def test_apply_rotary_emb_strided_positions(backend, heads):
    # Positions sliced out of a longer tensor keep a stride over 1.
    x, freqs_cis, _ = make(64, 64, heads=heads)
    pos = torch.randint(0, 4096, (128,), device="cuda", dtype=torch.int64)[::2]
    torch.testing.assert_close(
        load(backend)(x, freqs_cis, pos, False),
        reference(x, freqs_cis, pos, False),
        **TOLERANCE[x.dtype],
    )
    # The exact case the review reproduced.
    x = torch.ones((2, 2), device="cuda", dtype=torch.float32)
    f = torch.tensor(
        [[1 + 0j], [0 + 1j], [-1 + 0j]], device="cuda", dtype=torch.complex64
    )
    p = torch.tensor([0, 2, 1, 2], device="cuda", dtype=torch.int64)[::2]
    torch.testing.assert_close(
        load(backend)(x, f, p, False),
        reference(x, f, p, False),
        **TOLERANCE[torch.float32],
    )

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
OPERATOR = "fused_norm_rope_inplace"
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


def reference(kv, weight, eps, freqs_cis, positions):
    """The task's reference implementation, copied from its statement."""
    m, head_dim = kv.shape
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    rope_start = head_dim - rope_dim

    x = kv.to(torch.float32)
    rms_inv = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    normed = x * rms_inv
    if weight is not None:
        normed = normed * weight.to(torch.float32)

    rows = (
        torch.arange(m, device=kv.device)
        if positions is None
        else positions.long()
    )
    f = freqs_real[rows].to(torch.float32)
    f_real, f_imag = f[:, 0::2], f[:, 1::2]

    seg = normed[:, rope_start:]
    x_real, x_imag = seg[:, 0::2], seg[:, 1::2]
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    out = normed.clone()
    out[:, rope_start::2] = out_real
    out[:, rope_start + 1 :: 2] = out_imag
    return out.to(kv.dtype)


def make(
    m,
    head_dim,
    rope_dim,
    dtype,
    weight=True,
    positions=True,
    max_seq=4096,
    row_stride=None,
    seed=0,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = row_stride or head_dim
    storage = torch.randn(
        m, width, device="cuda", dtype=torch.float32, generator=generator
    )
    kv = storage.to(dtype)[:, :head_dim]
    w = (
        (
            1.0
            + 0.1
            * torch.randn(
                head_dim,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
        ).to(dtype)
        if weight
        else None
    )
    rows = max_seq if positions else m
    angle = torch.rand(
        rows, rope_dim // 2, device="cuda", generator=generator
    ) * (2 * torch.pi)
    freqs_cis = torch.polar(torch.ones_like(angle), angle)
    pos = (
        torch.randint(0, max_seq, (m,), device="cuda", generator=generator)
        if positions
        else None
    )
    return kv, w, 1e-6, freqs_cis, pos


def check(backend, *args, **kwargs):
    kv, w, eps, freqs_cis, pos = make(*args, **kwargs)
    before = kv.clone()
    expected = reference(kv, w, eps, freqs_cis, pos)
    actual = load(backend)(kv, w, eps, freqs_cis, pos)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.data_ptr() != kv.data_ptr()
    torch.testing.assert_close(actual, expected, **TOLERANCE[kv.dtype])
    # The reference returns a clone, so the input comes back untouched.
    assert torch.equal(kv, before)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("m", [1, 3, 64, 1000, 4096])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("positions", [True, False])
def test_fused_norm_rope_inplace_dsv4(backend, dtype, m, weight, positions):
    # DeepSeek-V4's KV prologue: a 512-wide row with a 64-wide rope tail.
    check(backend, m, 512, 64, dtype, weight=weight, positions=positions)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "head_dim, rope_dim",
    [(128, 64), (64, 64), (576, 64), (192, 64), (96, 32), (65, 64), (7, 2)],
)
@pytest.mark.parametrize("positions", [True, False])
def test_fused_norm_rope_inplace_widths(
    backend, head_dim, rope_dim, positions
):
    check(
        backend, 257, head_dim, rope_dim, torch.bfloat16, positions=positions
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_norm_rope_inplace_row_stride(backend):
    # A row view of a wider buffer, as when kv is sliced out of a projection.
    check(backend, 300, 512, 64, torch.bfloat16, row_stride=640)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_norm_rope_inplace_int32_positions(backend):
    kv, w, eps, freqs_cis, pos = make(129, 512, 64, torch.bfloat16)
    pos = pos.to(torch.int32)
    torch.testing.assert_close(
        load(backend)(kv, w, eps, freqs_cis, pos),
        reference(kv, w, eps, freqs_cis, pos),
        **TOLERANCE[kv.dtype],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_norm_rope_inplace_fp32_weight(backend):
    kv, w, eps, freqs_cis, pos = make(129, 512, 64, torch.bfloat16)
    w = w.float()
    torch.testing.assert_close(
        load(backend)(kv, w, eps, freqs_cis, pos),
        reference(kv, w, eps, freqs_cis, pos),
        **TOLERANCE[kv.dtype],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_norm_rope_inplace_empty(backend):
    kv, w, eps, freqs_cis, pos = make(0, 512, 64, torch.bfloat16)
    out = load(backend)(kv, w, eps, freqs_cis, pos)
    assert out.shape == (0, 512) and out.dtype == torch.bfloat16


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("m", [1, 7, 8, 74])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("positions", [True, False])
def test_fused_norm_rope_inplace_mla_rows(backend, m, weight, positions):
    # The graded cases include 576-wide rows (512 nope + 64 rope) at a few
    # rows each; the first Ascend kernel got their rope tail wrong.
    check(
        backend,
        m,
        576,
        64,
        torch.bfloat16,
        weight=weight,
        positions=positions,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_norm_rope_inplace_strided_table(backend):
    # A table whose complex entries are not adjacent, sliced from a wider
    # one; the reference indexes it the same way through view_as_real.
    kv, w, eps, freqs_cis, pos = make(129, 576, 64, torch.bfloat16)
    freqs_cis = torch.stack([freqs_cis, freqs_cis.conj()], dim=-1)[..., 0]
    assert freqs_cis.stride(-1) != 1
    torch.testing.assert_close(
        load(backend)(kv, w, eps, freqs_cis, pos),
        reference(kv, w, eps, freqs_cis, pos),
        **TOLERANCE[kv.dtype],
    )

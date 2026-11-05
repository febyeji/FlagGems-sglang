# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _pad(
    A,
    P,
    M: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x // K
    k = x % K
    value = tl.load(A + m * A0 + k * A1, (m < M) & (x < 16 * K), other=0)
    tl.store(P + x, value, x < 16 * K)


@triton.jit
def _gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.arange(0, 16)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + m[:, None] * K + k[None, :]
    bp = B + k[:, None] * B1 + n[None, :] * B0
    acc = tl.zeros((16, BN), tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        ap += BK
        bp += BK * B1
    tl.store(C + m[:, None] * N + n[None, :], acc, m[:, None] < M)


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = router_weights.shape[0]
    device = hidden_states.device
    bn = 64 if num_experts % 64 == 0 else 32 if num_experts % 32 == 0 else 16
    bk = 512 if hidden_dim % 512 == 0 else 256
    out = torch.empty(
        (num_tokens, num_experts), dtype=out_dtype, device=device
    )
    if num_tokens == 16 and hidden_states.stride() == (hidden_dim, 1):
        padded = hidden_states
    else:
        padded = torch.empty(
            (16, hidden_dim), dtype=hidden_states.dtype, device=device
        )
        _pad.run(
            hidden_states,
            padded,
            num_tokens,
            hidden_dim,
            hidden_states.stride(0),
            hidden_states.stride(1),
            4096,
            grid=(triton.cdiv(16 * hidden_dim, 4096),),
            warmup=False,
        )
    _gemm.run(
        padded,
        router_weights,
        out,
        num_tokens,
        num_experts,
        hidden_dim,
        router_weights.stride(0),
        router_weights.stride(1),
        bn,
        bk,
        grid=(triton.cdiv(num_experts, bn),),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["dsv3_router_gemm"]

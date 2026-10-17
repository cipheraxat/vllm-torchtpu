# Copyright 2026 Google LLC
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

"""Experimental TPU7x sparse GQA decode; no vLLM or PyTorch dependency."""

import jax

from .decode import qsa_decode_attention as _decode
from .eligibility import is_supported_by_fused_decode


def qsa_decode_attention(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    if not is_supported_by_fused_decode(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    ):
        raise NotImplementedError(
            "Fused decode requires TPU7x Q24/KV2/D256, ratio4, 512 groups, "
            "BF16 query/gate, native BF16/FP8 cache, positive ratio-aligned "
            "pages and valid nonempty int32 metadata; use RPA otherwise."
        )
    return _decode(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )

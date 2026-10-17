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

import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def is_supported_by_fused_decode(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
) -> bool:
    """Static eligibility; callers supply valid per-query selected groups."""
    if (
        q.ndim != 3
        or q.shape[1:] != (24, 256)
        or len(q) == 0
        or gate.shape != q.shape
        or q.dtype != jnp.bfloat16
        or gate.dtype != jnp.bfloat16
        or kv_heads != 2
        or ratio != 4
        or blocks.shape != (len(q), 512)
        or lengths.shape != (len(q),)
        or table.ndim != 2
        or table.shape[0] != len(q)
        or cache.ndim != 5
        or cache.shape[0] <= 0
        or cache.shape[1] <= 0
        or cache.shape[1] % ratio != 0
        or cache.shape[0] * cache.shape[1] > jnp.iinfo(jnp.int32).max
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table))
        or not isinstance(k_scale, (int, float))
        or not isinstance(v_scale, (int, float))
        or not math.isfinite(k_scale)
        or not math.isfinite(v_scale)
        or k_scale <= 0
        or v_scale <= 0
    ):
        return False
    packing = 32 // jax.dtypes.itemsize_bits(cache.dtype)
    # Physical token offsets, logical context lengths and flattened page-table
    # lookups use int32. These limits follow the indexing contract, not a
    # benchmark's largest batch or context.
    index_max = jnp.iinfo(jnp.int32).max
    if (
        cache.shape[2:] != (4 // packing, packing, 256)
        or table.shape[1] == 0
        or (table.shape[1] - 1) * cache.shape[1] >= index_max
        or table.size > index_max
    ):
        return False
    return pltpu.get_tpu_info().chip_version == pltpu.ChipVersion.TPU_7X

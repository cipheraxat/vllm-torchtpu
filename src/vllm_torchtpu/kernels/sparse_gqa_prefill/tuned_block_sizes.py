# SPDX-License-Identifier: Apache-2.0
"""TPU7x Q24/KV2/D256, ratio4, page1664 prefill tuning."""

import jax.numpy as jnp

# Both native dtypes use a 144 MiB compact selected-KV buffer.
SC_CHUNK_SIZE = {jnp.dtype(jnp.bfloat16): 32, jnp.dtype(jnp.float8_e4m3fn): 64}

# TPU7x: 16 words per reduction batch minimizes measured exact-union latency.
# At 8192 queries / 256K context, scratch stays below 65 MiB.
UNION_WORD_BATCH_SIZE = 16

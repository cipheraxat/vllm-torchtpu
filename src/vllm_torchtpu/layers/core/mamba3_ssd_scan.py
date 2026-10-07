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
"""JAX/Pallas hybrid implementation of Mamba-3 Selective State Space (SSD-3) scan."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Ensure highest precision for matmuls on TPU to match float32 references.
jax.config.update("jax_default_matmul_precision", "highest")


# ==============================================================================
# Pallas TPU Kernels (Native head batching, grid over chunks and heads)
# ==============================================================================


def mamba3_chunk_state_pallas(
    B_c: jnp.ndarray,
    decay_states: jnp.ndarray,
    X_c: jnp.ndarray,
) -> jnp.ndarray:
    """Pallas kernel for Mamba-3 chunk state accumulation.

    Computes:
      states = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states)

    Args:
      B_c: (n_chunks, n_heads, block_len, d_state) - Rotated & trapezoidal-scaled keys
      decay_states: (n_chunks, n_heads, block_len, 1) - exp(A_cumsum_last - A_cumsum)
      X_c: (n_chunks, n_heads, block_len, d_head) - Value tensor

    Returns:
      states: (n_chunks, n_heads, d_head, d_state)
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]

    def kernel(B_ref: Any, decay_ref: Any, X_ref: Any, out_ref: Any) -> None:
        B_val = jnp.squeeze(B_ref[...], axis=(0, 1)).astype(jnp.float32)
        decay_val = jnp.squeeze(decay_ref[...], axis=(0, 1)).astype(jnp.float32)
        X_val = jnp.squeeze(X_ref[...], axis=(0, 1)).astype(jnp.float32)

        X_scaled = X_val * decay_val  # (block_len, d_head)
        X_scaled_T = jnp.transpose(X_scaled, (1, 0))  # (d_head, block_len)

        out = jax.lax.dot_general(
            X_scaled_T,
            B_val,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out_ref[...] = jnp.expand_dims(out.astype(X_c.dtype), axis=(0, 1))

    grid = (n_chunks, n_heads)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, 1),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, d_head),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h: (c, h, 0, 0),
        block_shape=(1, 1, d_head, d_state),
    )

    out_shape = (n_chunks, n_heads, d_head, d_state)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, X_c.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True,
        ),
        interpret=(jax.default_backend() != "tpu"),
        name="mamba3_chunk_state_pallas",
    )(B_c, decay_states, X_c)


def mamba3_chunk_scan_pallas(
    C_c: jnp.ndarray,
    states: jnp.ndarray,
    decay_states: jnp.ndarray,
) -> jnp.ndarray:
    """Pallas kernel for Mamba-3 inter-chunk state-to-output projection.

    Computes:
      Y_off = jnp.einsum("chln,chpn,chl->chlp", C_c, states, state_decay_out)

    Args:
      C_c: (n_chunks, n_heads, block_len, d_state) - Rotated queries
      states: (n_chunks, n_heads, d_head, d_state) - Incoming chunk states
      decay_states: (n_chunks, n_heads, block_len, 1) - exp(A_cumsum)

    Returns:
      Y_off: (n_chunks, n_heads, block_len, d_head)
    """
    n_chunks, n_heads, block_len, d_state = C_c.shape
    d_head = states.shape[-2]

    def kernel(C_ref: Any, states_ref: Any, decay_ref: Any, out_ref: Any) -> None:
        C_val = jnp.squeeze(C_ref[...], axis=(0, 1)).astype(jnp.float32)
        states_val = jnp.squeeze(states_ref[...], axis=(0, 1)).astype(jnp.float32)
        decay_val = jnp.squeeze(decay_ref[...], axis=(0, 1)).astype(jnp.float32)

        states_T = jnp.transpose(states_val, (1, 0))  # (d_state, d_head)

        prod = jax.lax.dot_general(
            C_val,
            states_T,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

        out = (prod * decay_val).astype(C_c.dtype)
        out_ref[...] = jnp.expand_dims(out, axis=(0, 1))

    grid = (n_chunks, n_heads)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, d_head, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, 1),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h: (c, h, 0, 0),
        block_shape=(1, 1, block_len, d_head),
    )

    out_shape = (n_chunks, n_heads, block_len, d_head)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, C_c.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True,
        ),
        interpret=(jax.default_backend() != "tpu"),
        name="mamba3_chunk_scan_pallas",
    )(C_c, states, decay_states)


def _mamba3_fused_chunk_kernel(
    x_ref: Any,
    adt_ref: Any,
    b_ref: Any,
    c_ref: Any,
    init_state_ref: Any,
    y_ref: Any,
    final_state_ref: Any,
    s_ref: Any,
) -> None:
    """Fused intra-chunk strict causal scan and inter-chunk state Pallas kernel."""
    chunk_idx = pl.program_id(1)
    num_chunks = pl.num_programs(1)

    @pl.when(chunk_idx == 0)
    def _init() -> None:
        # init_state_ref has shape (1, 1, h_group, p, n); transpose to (h_group, n, p)
        s_ref[...] = jnp.swapaxes(init_state_ref[0, 0].astype(jnp.float32), -1, -2)

    x_val = x_ref[0]
    adt_val = adt_ref[0, :, :, 0]
    b_val = b_ref[0]
    c_val = c_ref[0]

    x = jnp.swapaxes(x_val, 0, 1)  # (h_group, chunk_size, p)
    adt = jnp.swapaxes(adt_val, 0, 1).astype(jnp.float32)  # (h_group, chunk_size)
    b = jnp.swapaxes(b_val, 0, 1)  # (h_group, chunk_size, n)
    c = jnp.swapaxes(c_val, 0, 1)  # (h_group, chunk_size, n)

    chunk_size = x.shape[1]
    row_idx = jnp.arange(chunk_size)[:, None]
    col_idx = jnp.arange(chunk_size)[None, :]
    tril_strict = row_idx > col_idx
    tril_incl = row_idx >= col_idx

    adt_expanded = jnp.where(tril_incl[None, :, :], adt[:, None, :], 0.0)
    da_cumsum = jnp.sum(adt_expanded, axis=-1)

    cb = jnp.matmul(c, jnp.swapaxes(b, -1, -2), preferred_element_type=jnp.float32)
    da_diff = da_cumsum[:, :, None] - da_cumsum[:, None, :]
    decay_mat = jnp.where(
        tril_strict[None, :, :],
        jnp.exp(jnp.clip(da_diff, -100.0, 0.0)),
        0.0,
    ).astype(x.dtype)
    intra_k = cb.astype(x.dtype) * decay_mat
    y_intra = jnp.matmul(intra_k, x, preferred_element_type=jnp.float32)

    da_end = da_cumsum[:, -1:]
    decay_to_end = jnp.exp(jnp.clip(da_end - da_cumsum, -100.0, 0.0)).astype(x.dtype)
    b_dec = b * decay_to_end[:, :, None]
    bx_dec = jnp.matmul(
        jnp.swapaxes(b_dec, -1, -2), x, preferred_element_type=jnp.float32
    )

    s = s_ref[...]
    decay_from_start = jnp.exp(jnp.clip(da_cumsum, -100.0, 0.0)).astype(x.dtype)
    c_dec = c * decay_from_start[:, :, None]
    y_inter = jnp.matmul(c_dec, s.astype(x.dtype), preferred_element_type=jnp.float32)

    y_val = (y_intra + y_inter).astype(x.dtype)
    chunk_total_decay = jnp.exp(jnp.clip(da_end[:, 0], -100.0, 0.0))
    s_new = s * chunk_total_decay[:, None, None] + bx_dec
    s_ref[...] = s_new

    y_ref[0] = jnp.swapaxes(y_val, 0, 1)

    @pl.when(chunk_idx == num_chunks - 1)
    def _save_final() -> None:
        final_state_ref[0, 0] = jnp.swapaxes(s_new, -1, -2)


def _run_fused_group_pallas(
    x_g: jnp.ndarray,
    adt_g: jnp.ndarray,
    k_g: jnp.ndarray,
    q_g: jnp.ndarray,
    init_s_g: jnp.ndarray,
    *,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs _mamba3_fused_chunk_kernel on a single head group."""
    b, seq_len, h_group, p = x_g.shape
    n = k_g.shape[-1]
    num_chunks = seq_len // chunk_size
    grid = (b, num_chunks)

    in_specs = [
        pl.BlockSpec((1, chunk_size, h_group, p), lambda b_i, c_i: (b_i, c_i, 0, 0)),
        pl.BlockSpec((1, chunk_size, h_group, 1), lambda b_i, c_i: (b_i, c_i, 0, 0)),
        pl.BlockSpec((1, chunk_size, h_group, n), lambda b_i, c_i: (b_i, c_i, 0, 0)),
        pl.BlockSpec((1, chunk_size, h_group, n), lambda b_i, c_i: (b_i, c_i, 0, 0)),
        pl.BlockSpec((1, 1, h_group, p, n), lambda b_i, c_i: (b_i, 0, 0, 0, 0)),
    ]
    out_specs = [
        pl.BlockSpec((1, chunk_size, h_group, p), lambda b_i, c_i: (b_i, c_i, 0, 0)),
        pl.BlockSpec((1, 1, h_group, p, n), lambda b_i, c_i: (b_i, 0, 0, 0, 0)),
    ]
    scratch_shapes = [pltpu.VMEM((h_group, n, p), jnp.float32)]

    y_g, final_s_g = pl.pallas_call(
        _mamba3_fused_chunk_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(x_g.shape, x_g.dtype),
            jax.ShapeDtypeStruct((b, 1, h_group, p, n), jnp.float32),
        ],
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=(jax.default_backend() != "tpu"),
        name="mamba3_fused_chunk_pallas",
    )(x_g, adt_g, k_g, q_g, init_s_g)
    return y_g, final_s_g[:, 0]


def mamba3_fused_chunk_pallas(
    x: jnp.ndarray,
    adt: jnp.ndarray,
    k_scaled: jnp.ndarray,
    q_rot: jnp.ndarray,
    initial_state: jnp.ndarray | None = None,
    *,
    chunk_size: int = 256,
    head_group: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Executes fused Pallas TPU Mamba-3 scan with head-group VMEM tiling.

    Args:
      x: (batch, seq_len, n_heads, d_head)
      adt: (batch, seq_len, n_heads)
      k_scaled: (batch, seq_len, n_heads, d_state)
      q_rot: (batch, seq_len, n_heads, d_state)
      initial_state: Optional (batch, n_heads, d_head, d_state)
      chunk_size: Chunk length dividing seq_len
      head_group: Number of heads per VMEM group slice (default 8 for TPU v7x)

    Returns:
      y: (batch, seq_len, n_heads, d_head)
      final_state: (batch, n_heads, d_head, d_state)
    """
    b, _, h, p = x.shape
    n = k_scaled.shape[-1]
    if initial_state is None:
        initial_state = jnp.zeros((b, h, p, n), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    init_expanded = initial_state[:, None, :, :, :]
    adt_expanded = adt[..., None]

    if h <= head_group or h % head_group != 0:
        return _run_fused_group_pallas(
            x,
            adt_expanded,
            k_scaled,
            q_rot,
            init_expanded,
            chunk_size=chunk_size,
        )

    y_groups = []
    s_groups = []
    for h_idx in range(0, h, head_group):
        h_slice = slice(h_idx, h_idx + head_group)
        y_g, s_g = _run_fused_group_pallas(
            x[:, :, h_slice, :],
            adt_expanded[:, :, h_slice, :],
            k_scaled[:, :, h_slice, :],
            q_rot[:, :, h_slice, :],
            init_expanded[:, :, h_slice, :, :],
            chunk_size=chunk_size,
        )
        y_groups.append(y_g)
        s_groups.append(s_g)

    return jnp.concatenate(y_groups, axis=2), jnp.concatenate(s_groups, axis=1)


# ==============================================================================
# Helper Functions
# ==============================================================================


def _segsum_strict_jax(x: jnp.ndarray, mimo_rank: int = 1) -> jnp.ndarray:
    """Computes strict causal segment sums (s < l) for Mamba-3 intra-chunk decay.

    Unlike Mamba-2 (which includes the diagonal s <= l inside the SSD scan),
    Mamba-3 uses a strictly lower-triangular intra-chunk mask (s < l) because
    the diagonal s == l contribution is governed by the trapezoidal weight
    gamma_t = dt_t * trap_t and added analytically outside the scan.
    """
    T = x.shape[-1]
    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

    mask_pos = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask_pos, x_repeated, 0.0)
    x_segsum = jnp.cumsum(x_masked, axis=-2)

    if mimo_rank > 1:
        n_tok = T // mimo_rank
        tok_mask = jnp.tril(jnp.ones((n_tok, n_tok), dtype=bool), k=-1)
        mask_strict = jnp.repeat(
            jnp.repeat(tok_mask, mimo_rank, axis=0), mimo_rank, axis=1
        )
    else:
        mask_strict = mask_pos

    return jnp.where(mask_strict, x_segsum, -jnp.inf)


# ==============================================================================
# Core Ragged Scan Operator
# ==============================================================================


def ssd3_minimal_discrete_hybrid(
    X_c: jnp.ndarray,
    A_c: jnp.ndarray,
    B_c: jnp.ndarray,
    C_c: jnp.ndarray,
    last_chunk_indices: jnp.ndarray,
    initial_states: jnp.ndarray | None = None,
    reset_mask: jnp.ndarray | None = None,
    mimo_rank: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Hybrid JAX/Pallas Mamba-3 Selective State Space (SSD-3) scan.

    Implements the strict-causal discrete SSD-3 scan over packed chunks using
    Pallas TPU kernels for chunk state accumulation and state-to-output
    conversion, and native JAX scan for ragged inter-chunk recurrence with
    sequence boundary resets.

    Args:
      X_c: (n_chunks, n_heads, block_len, d_head) - Packed value tensor
      A_c: (n_chunks, n_heads, block_len) - Log decay parameter (dt * A)
      B_c: (n_chunks, n_heads, block_len, d_state) - Rotated & trapezoidal-scaled keys
      C_c: (n_chunks, n_heads, block_len, d_state) - Rotated queries
      last_chunk_indices: (num_seqs,) - Maps each sequence to its final chunk
      initial_states: (n_chunks, n_heads, d_head, d_state) - Optional initial
        states placed at sequence start chunks
      reset_mask: (n_chunks,) - Boolean mask indicating sequence start chunks
      mimo_rank: MIMO channel rank R (1 for SISO, >1 when block_len = chunk_size * R)

    Returns:
      Y: (n_chunks, n_heads, block_len, d_head) - Scan output (strict causal s < l)
      final_states: (num_seqs, n_heads, d_head, d_state) - Final recurrent states
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]

    if initial_states is None:
        initial_states = jnp.zeros(
            (n_chunks, n_heads, d_head, d_state), dtype=jnp.float32
        )
    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool).at[0].set(True)

    A_c_f32 = A_c.astype(jnp.float32)

    if mimo_rank > 1:
        n_tok = block_len // mimo_rank
        X_r = X_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_head)
        B_r = B_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_state)
        C_r = C_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_state)
        if A_c_f32.shape[-1] == block_len:
            A_tok = jnp.sum(
                A_c_f32.reshape(n_chunks, n_heads, n_tok, mimo_rank), axis=-1
            )
        else:
            A_tok = A_c_f32
        A_cumsum = jnp.cumsum(A_tok, axis=-1)

        # 1. Intra-chunk strict lower-triangular output (s < l) with 64x64 token decay
        L_tok = jnp.exp(_segsum_strict_jax(A_tok, mimo_rank=1))
        cb = jnp.einsum(
            "chlrn,chsqn->chlsrq", C_r, B_r, preferred_element_type=jnp.float32
        )
        intra_k = (cb * L_tok[:, :, :, :, None, None]).astype(X_c.dtype)
        Y_diag = jnp.einsum(
            "chlsrq,chsqp->chlrp",
            intra_k,
            X_r,
            preferred_element_type=jnp.float32,
        ).reshape(n_chunks, n_heads, block_len, d_head)

        # 2. Intra-chunk state generation
        decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum).astype(X_c.dtype)
        X_scaled = X_r * decay_states[:, :, :, None, None]
        states_c = jnp.einsum(
            "chlrn,chlrp->chpn",
            B_r,
            X_scaled,
            preferred_element_type=jnp.float32,
        )
    else:
        A_cumsum = jnp.cumsum(A_c_f32, axis=-1)

        # 1. Intra-chunk strict lower-triangular output (s < l)
        L = jnp.exp(_segsum_strict_jax(A_c_f32, mimo_rank=1))
        cb = jnp.einsum("chln,chsn->chls", C_c, B_c, preferred_element_type=jnp.float32)
        intra_k = (cb * L).astype(X_c.dtype)
        Y_diag = jnp.einsum(
            "chls,chsp->chlp", intra_k, X_c, preferred_element_type=jnp.float32
        )

        # 2. Intra-chunk state generation
        decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum).astype(X_c.dtype)
        X_scaled = X_c * decay_states[..., None]
        states_c = jnp.einsum(
            "chln,chlp->chpn",
            B_c,
            X_scaled,
            preferred_element_type=jnp.float32,
        )

    # 3. Inter-chunk recurrence across ragged sequences (JAX Scan)
    decay_chunk = jnp.exp(A_cumsum[..., -1])

    def scan_body(
        carry: jnp.ndarray,
        args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        states_c_c, decay_chunk_c, init_state_c, reset_c = args
        incoming_state = jnp.where(reset_c, init_state_c.astype(jnp.float32), carry)
        decay_exp = decay_chunk_c[:, None, None]
        next_state = incoming_state * decay_exp + states_c_c.astype(jnp.float32)
        return next_state, (incoming_state, next_state)

    scan_args = (states_c, decay_chunk, initial_states, reset_mask)
    init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=jnp.float32)
    _, (states_incoming, states_outgoing) = jax.lax.scan(
        scan_body, init_carry, scan_args
    )

    # 4. Inter-chunk state to output conversion
    state_decay_out = jnp.exp(A_cumsum).astype(C_c.dtype)
    if mimo_rank > 1:
        C_scaled = C_r * state_decay_out[:, :, :, None, None]
        Y_off = jnp.einsum(
            "chlrn,chpn->chlrp",
            C_scaled,
            states_incoming.astype(C_c.dtype),
            preferred_element_type=jnp.float32,
        ).reshape(n_chunks, n_heads, block_len, d_head)
    else:
        C_scaled = C_c * state_decay_out[..., None]
        Y_off = jnp.einsum(
            "chln,chpn->chlp",
            C_scaled,
            states_incoming.astype(C_c.dtype),
            preferred_element_type=jnp.float32,
        )

    Y = (Y_diag + Y_off).astype(X_c.dtype)

    safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, n_chunks - 1)
    final_states = states_outgoing[safe_last_chunk_indices]
    return Y, final_states

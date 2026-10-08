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
"""Out-of-tree TPU custom operation and layer registration for MambaMixer3."""

import functools

import jax
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.layers.core.mamba_mixer3 import run_jax_mamba_mixer3
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import get_dp_size, get_mesh_shape_product


def mamba_mixer3_core_tpu(
    projected_states: jax.Array,
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    B_norm_weight: jax.Array,
    C_norm_weight: jax.Array,
    B_bias: jax.Array,
    C_bias: jax.Array,
    D: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array | None,
    psi: jax.Array | None,
    phi: jax.Array | None,
    zeta: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    chunk_size: int,
    is_mimo: bool,
    mimo_rank: int,
    A_floor: float,
    shard_groups: bool,
    rms_norm_eps: float,
    apply_rms_norm: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r = mimo_rank if is_mimo else 1
    target_aux_dim = r * (head_dim + d_state) + num_rope_angles
    aux_dim = aux_state.shape[-1]
    if aux_dim > target_aux_dim:
        aux_state_in = aux_state[..., :target_aux_dim]
    else:
        aux_state_in = aux_state

    (new_aux_state, new_recurrent_state), output = run_jax_mamba_mixer3(
        projected_states,
        aux_state_in,
        recurrent_state,
        B_norm_weight,
        C_norm_weight,
        B_bias,
        C_bias,
        D,
        dt_bias,
        norm_weight,
        psi,
        phi,
        zeta,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        mesh=mesh,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        num_rope_angles=num_rope_angles,
        chunk_size=chunk_size,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        A_floor=A_floor,
        shard_groups=shard_groups,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
    )
    if aux_dim > target_aux_dim:
        new_aux_state = aux_state.at[..., :target_aux_dim].set(new_aux_state)
    return new_aux_state, new_recurrent_state, output


@PluggableLayer.register("mamba_mixer3")
class MambaMixer3(MambaBase, PluggableLayer):
    """Mamba-3 Selective State Space Mixer layer with exponential-trapezoidal
    discretization and continuous-time 2D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        intermediate_size: int,
        use_bias: bool = False,
        n_groups: int = 1,
        num_heads: int = 80,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        use_rms_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        rope_fraction: float = 1.0,
        A_floor: float = 1e-4,
        chunk_size: int = 64,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % self.tp_size == 0, (
            "Tensor parallel world size must divide num_heads."
        )
        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide n_groups, "
            "then n_groups must equal 1."
        )

        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.rope_fraction = rope_fraction
        self.num_rope_angles = int((ssm_state_size // 2) * rope_fraction)
        self.A_floor = A_floor
        self.chunk_size = chunk_size
        self.rms_norm_eps = rms_norm_eps
        self.use_rms_norm = use_rms_norm
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tped_intermediate_size = divide(intermediate_size, self.tp_size)
        self.tped_dt_size = divide(num_heads, self.tp_size)
        self.tp_n_groups = (
            divide(n_groups, self.tp_size) if n_groups % self.tp_size == 0 else n_groups
        )
        self.groups_ssm_state_size = (
            self.mimo_rank * self.n_groups * self.ssm_state_size
        )
        self.tped_bc_size = self.mimo_rank * self.tp_n_groups * self.ssm_state_size
        self.aux_dim = (
            self.mimo_rank * (self.head_dim + self.ssm_state_size)
            + self.num_rope_angles
        )
        self.total_proj_size = (
            2 * intermediate_size
            + 2 * self.groups_ssm_state_size
            + 3 * num_heads
            + self.num_rope_angles
        )
        self.local_proj_size = (
            2 * self.tped_intermediate_size
            + 2 * self.tped_bc_size
            + 3 * self.tped_dt_size
            + self.num_rope_angles
        )

        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.local_proj_size * self.tp_size,
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        in_proj_loader = self._make_in_proj_weight_loader()
        self._set_param_weight_loader(self.in_proj.weight, in_proj_loader)
        if getattr(self.in_proj, "bias", None) is not None:
            self._set_param_weight_loader(self.in_proj.bias, in_proj_loader)

        self.B_norm_weight = nn.Parameter(torch.ones(ssm_state_size))
        self.C_norm_weight = nn.Parameter(torch.ones(ssm_state_size))

        head_loader = self._make_head_weight_loader(squeeze_siso_rank_dim=False)
        bc_bias_loader = self._make_head_weight_loader(squeeze_siso_rank_dim=True)

        if self.is_mimo:
            self.B_bias = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, ssm_state_size)
            )
            self.C_bias = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, ssm_state_size)
            )
            self.Psi = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim) / self.mimo_rank
            )
            self.Phi = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim) / self.mimo_rank
            )
            self.Zeta = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim)
            )
            set_weight_attrs(self.Psi, {"weight_loader": head_loader})
            set_weight_attrs(self.Phi, {"weight_loader": head_loader})
            set_weight_attrs(self.Zeta, {"weight_loader": head_loader})
        else:
            self.B_bias = nn.Parameter(torch.ones(self.tped_dt_size, ssm_state_size))
            self.C_bias = nn.Parameter(torch.ones(self.tped_dt_size, ssm_state_size))
            self.register_parameter("Psi", None)
            self.register_parameter("Phi", None)
            self.register_parameter("Zeta", None)

        self.D = nn.Parameter(torch.ones(self.tped_dt_size))
        self.dt_bias = nn.Parameter(torch.ones(self.tped_dt_size))

        set_weight_attrs(self.B_bias, {"weight_loader": bc_bias_loader})
        set_weight_attrs(self.C_bias, {"weight_loader": bc_bias_loader})
        set_weight_attrs(self.D, {"weight_loader": head_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": head_loader})

        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        try:
            vllm_config = get_current_vllm_config()
            compilation_config = vllm_config.compilation_config
            if prefix and prefix not in compilation_config.static_forward_context:
                compilation_config.static_forward_context[prefix] = self
        except Exception:
            pass

        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    @staticmethod
    def _set_param_weight_loader(param: nn.Parameter, loader) -> None:
        if isinstance(param, BasevLLMParameter):
            param.weight_loader = loader
        else:
            if hasattr(param, "weight_loader"):
                delattr(param, "weight_loader")
            set_weight_attrs(param, {"weight_loader": loader})

    def _make_in_proj_weight_loader(self):
        section_sizes = [
            self.intermediate_size,
            self.intermediate_size,
            self.groups_ssm_state_size,
            self.groups_ssm_state_size,
            self.num_heads,
            self.num_heads,
            self.num_heads,
            self.num_rope_angles,
        ]
        shard_flags = [
            True,
            True,
            (self.n_groups % self.tp_size == 0),
            (self.n_groups % self.tp_size == 0),
            True,
            True,
            True,
            False,
        ]
        tp_size = self.tp_size
        tp_rank = self.tp_rank

        def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.shape[0] == param.shape[0]:
                param.data.copy_(loaded_weight)
                return
            sections = torch.split(loaded_weight, section_sizes, dim=0)
            local_sections = []
            for sec, is_sharded in zip(sections, shard_flags):
                if is_sharded and tp_size > 1:
                    shard_len = sec.shape[0] // tp_size
                    local_sections.append(sec.narrow(0, tp_rank * shard_len, shard_len))
                else:
                    local_sections.append(sec)
            param.data.copy_(torch.cat(local_sections, dim=0))

        return _loader

    def _make_head_weight_loader(self, *, squeeze_siso_rank_dim: bool = False):
        tp_size = self.tp_size
        tp_rank = self.tp_rank

        def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if squeeze_siso_rank_dim:
                if (
                    loaded_weight.ndim == 3
                    and param.ndim == 2
                    and loaded_weight.shape[1] == 1
                ):
                    loaded_weight = loaded_weight.squeeze(1)
                elif (
                    loaded_weight.ndim == 2 and param.ndim == 3 and param.shape[1] == 1
                ):
                    loaded_weight = loaded_weight.unsqueeze(1)
            if loaded_weight.shape == param.shape:
                param.data.copy_(loaded_weight)
                return
            shard_size = param.shape[0]
            assert loaded_weight.shape[0] == shard_size * tp_size
            param.data.copy_(loaded_weight.narrow(0, tp_rank * shard_size, shard_size))

        return _loader

    @property
    def mimo_x(self) -> nn.Parameter | None:
        return self.Psi

    @property
    def mimo_o(self) -> nn.Parameter | None:
        return self.Phi

    @property
    def mimo_z(self) -> nn.Parameter | None:
        return self.Zeta

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is not None and self.cache_config is not None:
            _, ssm_state_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )
            return torch.float32, ssm_state_dtype
        return torch.float32, torch.float32

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        aux_shape = (self.tped_dt_size, self.aux_dim)
        ssm_shape = (self.tped_dt_size, self.head_dim, self.ssm_state_size)
        return aux_shape, ssm_shape

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA2


@MambaMixer3.register_oot
class VllmMambaMixer3(MambaMixer3):
    """Out-of-tree TPU implementation of MambaMixer3 backed by Pallas/JAX."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._apply_py_norm = False
        self.mamba_op = self._build_mamba_op()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        _, ssm_state_dtype = super().get_state_dtype()
        return torch.float32, ssm_state_dtype

    def _use_fused_jax_rms_norm(self, jax_tp_size: int) -> bool:
        if getattr(self.norm, "weight", None) is None:
            return True
        orig_single_group_replicated = (
            self.tp_size > 1
            and jax_tp_size == 1
            and (self.n_groups // self.tp_size) == 1
            and getattr(self.norm, "n_groups", self.n_groups) == 1
        )
        return not orig_single_group_replicated

    def _build_mamba_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh
        parallel_config = getattr(vllm_context.vllm_config, "parallel_config", None)
        dp_enabled = parallel_config is not None and get_dp_size(parallel_config) > 1

        head_axis = "model" if "model" in mesh.axis_names else None
        jax_tp_size = get_mesh_shape_product(mesh, head_axis) if head_axis else 1

        is_mimo = getattr(self, "is_mimo", False)
        mimo_rank = getattr(self, "mimo_rank", 1) if is_mimo else 1
        num_rope_angles = getattr(self, "num_rope_angles", self.ssm_state_size // 2)

        tp_n_groups = getattr(
            self,
            "tp_n_groups",
            self.n_groups // self.tp_size
            if self.n_groups % self.tp_size == 0
            else self.n_groups,
        )
        if tp_n_groups % jax_tp_size == 0 and jax_tp_size == 1:
            shard_groups = True
            local_n_groups = tp_n_groups // jax_tp_size
        else:
            shard_groups = False
            local_n_groups = tp_n_groups

        local_num_heads = (self.num_heads // self.tp_size) // jax_tp_size
        chunk_size = getattr(self, "chunk_size", 32 if dp_enabled else 64)
        rms_norm_eps = getattr(self.norm, "variance_epsilon", 1e-5)
        apply_rms_norm = self._use_fused_jax_rms_norm(jax_tp_size)
        self._apply_py_norm = not apply_rms_norm
        A_floor = getattr(self, "A_floor", 1e-4)

        op_name = f"pallas::mamba_mixer3_{self.prefix.replace('.', '_')}"
        wrapped_fn = functools.partial(
            mamba_mixer3_core_tpu,
            mesh=mesh,
            num_heads=local_num_heads,
            head_dim=self.head_dim,
            n_groups=local_n_groups,
            d_state=self.ssm_state_size,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            A_floor=A_floor,
            shard_groups=shard_groups,
            rms_norm_eps=rms_norm_eps,
            apply_rms_norm=apply_rms_norm,
        )

        mamba_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=(1, 2),
        )

        out_dim = (self.num_heads // self.tp_size) * self.head_dim

        def _fake_mamba3(
            projected_states: torch.Tensor,
            aux_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            *args,
            **kwargs,
        ):
            num_tokens = projected_states.size(0)
            return (
                torch.empty_like(aux_state),
                torch.empty_like(recurrent_state),
                torch.empty(
                    (num_tokens, out_dim),
                    dtype=projected_states.dtype,
                    device=projected_states.device,
                ),
            )

        mamba_jax_op.register_fake(_fake_mamba3)

        def mamba3_impl(
            projected_states: torch.Tensor,
            aux_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            B_norm_weight: torch.Tensor,
            C_norm_weight: torch.Tensor,
            B_bias: torch.Tensor,
            C_bias: torch.Tensor,
            D: torch.Tensor,
            dt_bias: torch.Tensor,
            norm_weight: torch.Tensor | None,
            psi: torch.Tensor | None,
            phi: torch.Tensor | None,
            zeta: torch.Tensor | None,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
        ) -> torch.Tensor:
            new_aux, new_rec, outputs = mamba_jax_op(
                projected_states,
                aux_state,
                recurrent_state,
                B_norm_weight,
                C_norm_weight,
                B_bias,
                C_bias,
                D,
                dt_bias,
                norm_weight,
                psi,
                phi,
                zeta,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
            )

            aux_state.copy_(new_aux)
            recurrent_state.copy_(new_rec)
            return outputs

        return mamba3_impl

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        out_dim = (self.num_heads // self.tp_size) * self.head_dim

        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is None or kv_cache[0].numel() == 0:
            ssm_output = torch.zeros(
                (num_tokens, out_dim),
                dtype=projected_states.dtype,
                device=projected_states.device,
            )
            apply_py_norm = True
        else:
            fc = get_forward_context()
            attn_metadata = fc.attn_metadata[self.prefix]
            aux_state, recurrent_state = kv_cache

            if attn_metadata.mamba_state_indices is not None:
                state_indices = attn_metadata.mamba_state_indices.to(torch.int32)
            else:
                max_reqs = attn_metadata.seq_lens.shape[0]
                max_blocks_per_req = attn_metadata.block_tables.shape[0] // max_reqs
                block_tables_2d = torch.reshape(
                    attn_metadata.block_tables,
                    (max_reqs, max_blocks_per_req),
                )
                state_indices = block_tables_2d[:, 0].to(torch.int32)

            request_distribution = attn_metadata.request_distribution
            mamba_request_distribution = getattr(
                attn_metadata, "mamba_request_distribution", None
            )
            if mamba_request_distribution is not None:
                request_distribution = mamba_request_distribution

            apply_py_norm = getattr(self, "_apply_py_norm", False)
            norm_weight = getattr(self.norm, "weight", None)

            ssm_output = self.mamba_op(
                projected_states,
                aux_state,
                recurrent_state,
                self.B_norm_weight,
                self.C_norm_weight,
                self.B_bias,
                self.C_bias,
                self.D,
                self.dt_bias,
                norm_weight,
                getattr(self, "Psi", None),
                getattr(self, "Phi", None),
                getattr(self, "Zeta", None),
                state_indices,
                attn_metadata.query_start_loc,
                request_distribution,
                attn_metadata.seq_lens,
            )

        if (
            apply_py_norm
            and not getattr(self, "is_mimo", False)
            and getattr(self.norm, "weight", None) is not None
        ):
            gate = projected_states[..., : self.tped_intermediate_size]
            hidden_states = self.norm(ssm_output, gate)
        elif apply_py_norm and not getattr(self, "is_mimo", False):
            gate = projected_states[..., : self.tped_intermediate_size]
            hidden_states = ssm_output * torch.nn.functional.silu(gate)
        else:
            hidden_states = ssm_output

        output, _ = self.out_proj(hidden_states)
        return output

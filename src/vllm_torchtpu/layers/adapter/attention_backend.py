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
"""The TPU attention backends that --attention-backend <name> selects.

register_attention_backend adds a backend under a name: the name joins vLLM's
AttentionBackendEnum (vllm#41091), so --attention-backend <name> passes vLLM's
check, and TpuPlatform.get_attn_backend_cls resolves it to the backend's vLLM
attention backend class instead of overriding it to FLASH_ATTN. Its
TPUAttentionBackend checks the engine configuration before any worker starts.
With no registered name selected the layers run RPA (FLASH_ATTN), or batched
RPA (CUSTOM).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from vllm import envs as vllm_envs
from vllm.ray.lazy_utils import is_in_ray_actor
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class TPUAttentionBackend:
    """The runner side of a TPU attention backend."""

    # The --attention-backend name, lower case.
    name: ClassVar[str]

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Raise for an engine configuration this backend cannot serve. It may
        adjust the configuration, e.g. the block size. Runs in
        TpuPlatform.check_and_update_config, before any worker starts."""


class RPAAttentionBackend(TPUAttentionBackend):
    """The default: RPA v3 (FLASH_ATTN) or batched RPA (CUSTOM)."""

    name = "rpa"


@dataclass(frozen=True)
class _RegisteredBackend:
    attention: Callable[[], TPUAttentionBackend]
    vllm_backend: str


_REGISTERED_BACKENDS: dict[str, _RegisteredBackend] = {}


def register_attention_backend(
    name: str, *, attention: Callable[[], TPUAttentionBackend], vllm_backend: str
) -> None:
    """Make --attention-backend <name> run these backends.

    vllm_backend is the class path of the vLLM attention backend the attention
    layers get, a PallasAttentionBackend subclass. attention makes its
    TPUAttentionBackend.

    Call it from a vllm.general_plugins entry point, which vLLM runs in every
    process before it parses --attention-backend, and may run more than once
    in one process."""
    name = name.lower()
    member = AttentionBackendEnum.__members__.get(name.upper())
    if member is None:
        AttentionBackendEnum.register(name.upper(), vllm_backend)
    elif member.get_path() != vllm_backend:
        raise ValueError(
            f"vLLM already has an attention backend named {name.upper()}, at "
            f"{member.get_path()}."
        )
    _REGISTERED_BACKENDS[name] = _RegisteredBackend(attention, vllm_backend)


def is_registered_attention_backend(backend: AttentionBackendEnum | None) -> bool:
    """Whether vLLM's backend is one of register_attention_backend."""
    return backend is not None and backend.name.lower() in _REGISTERED_BACKENDS


def attention_backend_name(vllm_config: "VllmConfig") -> str:
    """The backend --attention-backend selects: a registered name, else "rpa"
    (FLASH_ATTN, CUSTOM, no flag, or a backend vLLM picked itself)."""
    backend = vllm_config.attention_config.backend
    if is_registered_attention_backend(backend):
        assert backend is not None
        return backend.name.lower()
    return RPAAttentionBackend.name


def create_attention_backend(name: str) -> TPUAttentionBackend:
    """The runner's TPUAttentionBackend of that name."""
    if name == RPAAttentionBackend.name:
        return RPAAttentionBackend()
    return _REGISTERED_BACKENDS[name].attention()


def check_attention_backend_config(vllm_config: "VllmConfig") -> None:
    """Raise if a registered backend is selected for an engine whose processes
    cannot rebuild its name, or that the backend's check_config rejects. Runs
    in TpuPlatform.check_and_update_config after the executor is chosen.

    register_attention_backend adds the name to AttentionBackendEnum at
    runtime. Enum members pickle by value, and a process that unpickles
    vllm_config before it loads the plugins, a spawned engine core or worker
    or a Ray worker, cannot rebuild the member. A forked process inherits it,
    and an engine that starts no process pickles nothing. vLLM forces spawn
    later for a Ray actor or --numa-bind, so both are rejected here too.
    Spawned API servers (--api-server-count > 1) get the unparsed string and
    are fine.

    MLA models are rejected here too, so the API server fails before any
    worker starts; get_attn_backend_cls keeps the same check per layer.

    TODO: support spawn and Ray. vLLM must rebuild a registered
    AttentionBackendEnum member when it unpickles it."""
    name = attention_backend_name(vllm_config)
    if name == RPAAttentionBackend.name:
        return
    parallel_config = vllm_config.parallel_config
    starts_processes = (
        vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING
        or parallel_config.distributed_executor_backend != "uni"
    )
    model_config = vllm_config.model_config
    if model_config is not None and model_config.use_mla:
        raise ValueError(
            f"--attention-backend {name} does not support MLA models; leave "
            "--attention-backend unset."
        )
    reasons = []
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        reasons.append("TPU_MULTIHOST_BACKEND=ray")
    if is_in_ray_actor():
        reasons.append("the engine runs in a Ray actor")
    if starts_processes and vllm_envs.VLLM_WORKER_MULTIPROC_METHOD == "spawn":
        reasons.append(
            "VLLM_WORKER_MULTIPROC_METHOD=spawn (vllm serve sets it when unset; "
            "set VLLM_WORKER_MULTIPROC_METHOD=fork)"
        )
    if starts_processes and parallel_config.numa_bind:
        reasons.append("--numa-bind, which forces spawn")
    if reasons:
        raise ValueError(
            f"--attention-backend {name} supports a single host with forked "
            f"processes only, not: {', '.join(reasons)}."
        )
    create_attention_backend(name).check_config(vllm_config)

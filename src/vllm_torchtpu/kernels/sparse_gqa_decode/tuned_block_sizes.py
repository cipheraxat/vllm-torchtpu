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

"""TPU7x Q24/KV2/D256 decode configuration from the standalone sweep."""

from vllm_torchtpu.kernels.experimental.batched_rpa import configs

DECODE_BLOCK = configs.BlockSizes(
    bq_sz=1,
    bq_c_sz=1,
    bkv_sz=256,
    batch_size=8,
)

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
"""Checked-in named execution profiles.

`default` holds the `default` profile's model-agnostic tile rule (imported
eagerly; it depends only on `exec_spec`). Named profiles register themselves when
their module is imported; `profile.get_profile` / `list_profiles` call
`load_builtin_profiles()` first. A model's default profile lives with the model
and registers when the model is imported.
"""

from __future__ import annotations

from . import default


def load_builtin_profiles() -> None:
    """Import (and thereby register) every checked-in named profile."""


__all__ = ["default", "load_builtin_profiles"]

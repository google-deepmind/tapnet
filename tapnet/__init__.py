# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Legacy API for TAP.  Prefer importing from project subfolders."""

import importlib

_LAZY_SUBMODULES = {
    "tapir_model": ".models.tapir_model",
    "tapnet_model": ".models.tapnet_model",
    "tapir_clustering": ".robotap.tapir_clustering",
    "evaluation_datasets": ".tapvid.evaluation_datasets",
}


def __getattr__(name):  # pylint: disable=invalid-name
  if name in _LAZY_SUBMODULES:
    module = importlib.import_module(_LAZY_SUBMODULES[name], __package__)
    globals()[name] = module
    return module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pylint: disable=invalid-name
  return sorted(set(globals()) | set(_LAZY_SUBMODULES))

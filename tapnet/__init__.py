# Copyright 2025 Google LLC
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

from tapnet.models import tapir_model  # pylint:disable=g-importing-member
from tapnet.models import tapnet_model  # pylint:disable=g-importing-member
from tapnet.robotap import tapir_clustering  # pylint:disable=g-importing-member
from tapnet.tapvid import evaluation_datasets  # pylint:disable=g-importing-member

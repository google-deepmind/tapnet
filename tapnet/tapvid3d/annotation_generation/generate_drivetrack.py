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

"""Download the Waymo Open + DriveTrack videos and annotations.

The *.npz files in the GCS bucket for the DriveTrack split contains both
the preprocessed annotations and source videos, so all we need to do is
bulk download them to the local machine.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from tapnet.tapvid3d.annotation_generation import gcs_utils


_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "tapvid3d_dataset/drivetrack/",
    "Path to folder to store output npz files containing all fields.",
)

_DEBUG = flags.DEFINE_boolean(
    "debug",
    False,
    "Whether to run in debug mode, downloads only one video.",
)

_SPLIT = flags.DEFINE_enum(
    "split",
    "all",
    ["minival", "full_eval", "all"],
    """
    If True, compute metrics on the minival split;
    otherwise uses the full_eval split.
    """,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(_OUTPUT_DIR.value):
    os.makedirs(_OUTPUT_DIR.value)

  gcs_utils.download_tapvid3d_files(
      _OUTPUT_DIR.value, _SPLIT.value, "drivetrack", _DEBUG.value
  )


if __name__ == "__main__":
  app.run(main)

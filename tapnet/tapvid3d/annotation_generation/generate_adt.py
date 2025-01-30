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

"""Download the ADT query points and compute the remaining annotations.

The *.npz files in the GCS bucket for the ADT split contain only the query
points. This script loads the ADT scenes from the GCS bucket, computes the
remaining annotations and saves them to *.npz files in the local output folder.
"""

import collections
import glob
import os
from typing import Sequence

from absl import app
from absl import flags
from tapnet.tapvid3d.annotation_generation import adt_utils
from tapnet.tapvid3d.annotation_generation import gcs_utils
import tqdm


_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "tapvid3d_dataset/adt/",
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

_ADT_BASE_PATH = flags.DEFINE_string(
    "adt_base_path",
    "",
    "Path to folder containing ADT scenes as subfolders.",
)


def generate_adt_npz(
    adt_base_path: str, input_npz_dir: str, output_npz_dir: str
):
  """Generates the final ADT npz files, adding the remaining annotations."""
  input_npz = sorted(glob.glob(os.path.join(input_npz_dir, "*.npz")))
  done_npz = [
      os.path.basename(x)
      for x in sorted(glob.glob(os.path.join(output_npz_dir, "*.npz")))
  ]

  # Filter completed files.
  input_npz = list(
      filter(lambda x: os.path.basename(x) not in done_npz, input_npz)
  )

  # Group pending files by video.
  pending_vid_chunks = collections.defaultdict(list)
  for f in input_npz:
    basename = os.path.basename(f)[:-4].split("_")
    vid = "_".join(basename[:-1])
    chunk = int(basename[-1])
    pending_vid_chunks[vid].append(chunk)

  for vid, chunks in tqdm.tqdm(
      pending_vid_chunks.items(), total=len(pending_vid_chunks)
  ):
    adt_utils.process_vid(
        adt_base_path,
        input_npz_dir,
        output_npz_dir,
        vid,
        chunks,
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  tmp_adt_dir = os.path.join(_OUTPUT_DIR.value, "tmp")

  # Download ADT npz's containing query points only.
  gcs_utils.download_tapvid3d_files(
      tmp_adt_dir, _SPLIT.value, "adt", _DEBUG.value
  )

  # Compute the remaining annotations and save them to *.npz files.
  generate_adt_npz(_ADT_BASE_PATH.value, tmp_adt_dir, _OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)

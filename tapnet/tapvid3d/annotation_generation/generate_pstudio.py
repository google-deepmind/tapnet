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

"""Download the PStudio annotations and join with video data.

The *.npz files in the GCS bucket for the ADT split contain all annotation but
don't contain the video data. This script downloads the video data and adds it
to the *.npz files.
"""

# pylint: disable=g-import-not-at-top,g-bad-import-order
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import glob
import os
from typing import Sequence
import urllib.request
import zipfile

from absl import app
from absl import flags
from tapnet.tapvid3d.annotation_generation import gcs_utils
import PIL.Image
import numpy as np
import tqdm


_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "tapvid3d_dataset/pstudio/",
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

_PSTUDIO_DATA = flags.DEFINE_string(
    "pstudio_url",
    "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip",
    "URL of PStudio data.",
)


def generate_pstudio_npz(
    pstudio_base_path: str, input_npz_dir: str, output_npz_dir: str
):
  """Generates the final PStudio npz files, adding the video data."""
  input_npz = sorted(glob.glob(os.path.join(input_npz_dir, "*.npz")))
  done_npz = [
      os.path.basename(x)
      for x in sorted(glob.glob(os.path.join(output_npz_dir, "*.npz")))
  ]

  # Filter completed files.
  input_npz = list(
      filter(lambda x: os.path.basename(x) not in done_npz, input_npz)
  )

  # For each example, load the video data and add it to the npz file.
  for filename in tqdm.tqdm(input_npz):
    example = dict(np.load(filename, allow_pickle=True))
    out_fn = os.path.join(output_npz_dir, os.path.basename(filename))
    seq, cam_id = os.path.basename(filename)[:-4].split("_")
    # load rgb images
    im_fns = sorted(
        glob.glob(os.path.join(pstudio_base_path, f"{seq}/ims/{cam_id}/*.jpg"))
    )
    ims = [np.array(PIL.Image.open(im_fn)) for im_fn in im_fns]
    ims_jpeg = [np.array(tf.io.encode_jpeg(im)).item() for im in ims]
    example["images_jpeg_bytes"] = ims_jpeg
    np.savez(out_fn, **example)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  tmp_pstudio_dir = os.path.join(_OUTPUT_DIR.value, "tmp")

  gcs_utils.download_tapvid3d_files(
      tmp_pstudio_dir, _SPLIT.value, "pstudio", _DEBUG.value
  )
  # Download and extract PStudio video data.
  pstudio_zip = os.path.join(tmp_pstudio_dir, "data.zip")
  pstudio_data = os.path.join(tmp_pstudio_dir, "data")
  if not os.path.exists(pstudio_zip):
    print(f"Downloading PStudio data to {pstudio_zip}")
    urllib.request.urlretrieve(_PSTUDIO_DATA.value, pstudio_zip)
  else:
    print(f"Skipping download, PStudio data already exists: {pstudio_zip}")
  if not os.path.exists(pstudio_data):
    print(f"Extracting PStudio data to {pstudio_data}")
    with zipfile.ZipFile(pstudio_zip, "r") as zip_file:
      zip_file.extractall(tmp_pstudio_dir)
  else:
    print(f"Skipping extraction, PStudio data already exists: {pstudio_data}")

  # Compute the remaining annotations and save them to *.npz files.
  generate_pstudio_npz(pstudio_data, tmp_pstudio_dir, _OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)

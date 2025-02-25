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

"""Utils for downloading TAPVid3d data from GCS."""

import os
import sys
from absl import logging
import requests
import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, top_level_dir)
from tapnet.tapvid3d.splits import tapvid3d_splits  # pylint: disable=g-import-not-at-top, g-bad-import-order

TAPVID3D_GCS_URL = (
    "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/v1.0"
)


def download_tapvid3d_files(
    output_dir: str, split: str, subset: str, debug: bool
):
  """Downloads files from the given split and subset."""
  os.makedirs(output_dir, exist_ok=True)
  if split == "minival":
    filenames_to_download = tapvid3d_splits.get_minival_files(subset)
  elif split == "full_eval":
    filenames_to_download = tapvid3d_splits.get_full_eval_files(subset)
  elif split == "all":
    filenames_to_download = tapvid3d_splits.get_all_files(subset)
  else:
    raise ValueError(f"Unknown split: {split}")

  logging.info("Downloading %s split", split)
  for filename in tqdm.tqdm(
      filenames_to_download, total=len(filenames_to_download)
  ):
    local_path = os.path.join(output_dir, filename)
    gcs_url = get_tapvid3d_gcs_urls(
        filenames=filename,
        url_postfix=subset,
    )
    logging.info("Downloading %s to %s", gcs_url, local_path)
    download_file_from_url(input_url=gcs_url, output_filepath=local_path)
    if debug:
      logging.info("Stopping after one video, debug run.")
      break
  logging.info("Finished downloading all examples!")


def download_file_from_url(input_url: str, output_filepath: str):
  """Download the GCS file from the given URL to the given output path."""
  if os.path.exists(output_filepath):
    logging.info("Skipping download, file already exists: %s", output_filepath)
    return
  response = requests.get(input_url, stream=True)
  if response.status_code == 200:
    logging.info("Downloading: %s to %s", input_url, output_filepath)
    with open(output_filepath, "wb") as f:
      for chunk in response.iter_content(chunk_size=1024):
        f.write(chunk)
    logging.info("Downloaded!")
  else:
    logging.info("Download failed. HTTP Status Code: %d", response.status_code)


def get_tapvid3d_gcs_urls(
    filenames: str | list[str], url_postfix: str = ""
) -> str | list[str]:
  if isinstance(filenames, str):
    return f"{TAPVID3D_GCS_URL}/{url_postfix}/{filenames}"
  else:
    return [
        f"{TAPVID3D_GCS_URL}/{url_postfix}/{filename}" for filename in filenames
    ]

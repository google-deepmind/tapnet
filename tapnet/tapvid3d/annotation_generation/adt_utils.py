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

"""Utilities for generating TAPVid3d ADT npz files."""

# pylint: disable=g-import-not-at-top,g-bad-import-order
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import os

import numpy as np
import numpy.typing as npt
from PIL import Image
from projectaria_tools.core import calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider

import tqdm
from tapnet.tapvid3d.annotation_generation import adt_v1v2_mappings


# Fixed hyperparameters for generating the ADT data.
N_FRAMES = 300
HEIGHT = 512
WIDTH = 512
FOCAL_LENGTH = 280


class ADTVideoProcessor:
  """ADT video processor."""

  def __init__(self, sequence_path: str):
    self.timestamps_ns, self.gt_provider, self.stream_id = self._load_adt_data(
        sequence_path
    )

  def _load_adt_data(self, sequence_path: str):
    """Loads ADT data for a given sequence."""
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(
        selected_device_number, False
    )
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    stream_id = StreamId("214-1")
    timestamps_ns = np.array(
        gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
    )
    # Remove timestamps without annotations.
    timestamps_ns = timestamps_ns[
        timestamps_ns > gt_provider.get_start_time_ns()
    ]
    timestamps_ns = timestamps_ns[timestamps_ns < gt_provider.get_end_time_ns()]
    return timestamps_ns, gt_provider, stream_id

  def extract_image_data(
      self, chunk_timestamps_ns: list[int]
  ) -> tuple[
      list[npt.NDArray], list[npt.NDArray], list[npt.NDArray], list[int]
  ]:
    """Extracts image, depth and segmentation data for a given video chunk."""
    sensor_name = (
        self.gt_provider.raw_data_provider_ptr().get_label_from_stream_id(
            self.stream_id
        )
    )
    device_calib = (
        self.gt_provider.raw_data_provider_ptr().get_device_calibration()
    )
    src_calib = device_calib.get_camera_calib(sensor_name)
    identity_tnf = calibration.get_linear_camera_calibration(
        1, 1, 1
    ).get_transform_device_camera()
    dst_calib = calibration.CameraCalibration(
        "camera-rgb",
        calibration.CameraModelType.LINEAR,
        np.array([FOCAL_LENGTH, FOCAL_LENGTH, WIDTH / 2, HEIGHT / 2]),
        identity_tnf,
        WIDTH,
        HEIGHT,
        None,
        np.pi,
        "LinearCameraCalibration",
    )
    rgb_ims = []
    depth_ims = []
    segmentation_ims = []
    ok_timestamps_ns = []
    for select_timestamps_ns in chunk_timestamps_ns:
      depth_with_dt = self.gt_provider.get_depth_image_by_timestamp_ns(
          select_timestamps_ns, self.stream_id
      )
      segmentation_with_dt = (
          self.gt_provider.get_segmentation_image_by_timestamp_ns(
              select_timestamps_ns, self.stream_id
          )
      )
      image_with_dt = self.gt_provider.get_aria_image_by_timestamp_ns(
          select_timestamps_ns, self.stream_id
      )
      # Check if the result is valid.
      if (
          image_with_dt.is_valid()
          and depth_with_dt.is_valid()
          and segmentation_with_dt.is_valid()
      ):
        ok_timestamps_ns.append(select_timestamps_ns)
      else:
        continue
      image = image_with_dt.data().to_numpy_array()
      image = calibration.distort_by_calibration(image, dst_calib, src_calib)
      rgb_ims.append(image)

      depth_image = depth_with_dt.data().to_numpy_array()
      depth_image = calibration.distort_depth_by_calibration(
          depth_image, dst_calib, src_calib
      )
      depth_ims.append(depth_image)

      segmentation_data = segmentation_with_dt.data().to_numpy_array()
      segmentation_data = calibration.distort_label_by_calibration(
          segmentation_data, dst_calib, src_calib
      )
      segmentation_ims.append(segmentation_data)
    chunk_timestamps_ns = ok_timestamps_ns
    return rgb_ims, depth_ims, segmentation_ims, chunk_timestamps_ns


def process_vid(
    input_adt_path: str,
    input_npz_path: str,
    output_npz_path: str,
    seq_name: str,
    chunks: list[int],
):
  """Processes multiple chunks of a single video."""
  adt_v2_name = adt_v1v2_mappings.ADT_MAPPINGS[seq_name]
  sequence_path = os.path.join(input_adt_path, adt_v2_name)
  adt_processor = ADTVideoProcessor(sequence_path)

  for chunk_idx in tqdm.tqdm(chunks):
    track_fn = os.path.join(output_npz_path, f"{seq_name}_{chunk_idx}.npz")
    chunk_timestamps_ns = adt_processor.timestamps_ns[
        chunk_idx * N_FRAMES : (chunk_idx + 1) * N_FRAMES
    ]

    rgb_ims, _, _, _ = adt_processor.extract_image_data(chunk_timestamps_ns)
    rgb_ims = [np.array(Image.fromarray(im).rotate(-90)) for im in rgb_ims]
    rgb_jpegs = [np.array(tf.io.encode_jpeg(im)).item() for im in rgb_ims]

    # Load query points.
    in_npz = np.load(
        os.path.join(input_npz_path, f"{seq_name}_{chunk_idx}.npz"),
        allow_pickle=True,
    )
    queries_xyt = in_npz["queries_xyt"]
    trajectories = in_npz["tracks_XYZ"]
    visibilities = in_npz["visibility"]

    # Verify video means.
    video_means = np.stack([np.mean(x, axis=(0, 1)) for x in rgb_ims], axis=0)
    assert np.allclose(video_means, in_npz["video_means"], atol=1e-3)

    example = {
        "images_jpeg_bytes": rgb_jpegs,
        "queries_xyt": queries_xyt,
        "tracks_XYZ": trajectories,
        "visibility": visibilities,
        "fx_fy_cx_cy": np.array(
            [FOCAL_LENGTH, FOCAL_LENGTH, WIDTH / 2, HEIGHT / 2]
        ),
    }
    np.savez(track_fn, **example)

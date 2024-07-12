# Copyright 2024 DeepMind Technologies Limited
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
import glob
import hashlib

import numpy as np
import numpy.typing as npt
from PIL import Image
from projectaria_tools.core import calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider

import torch
from torchvision.models import segmentation as torch_seg
import tqdm
from visu3d.math import interp_utils


# Fixed hyperparameters for generating the ADT data.
N_FRAMES = 300
HEIGHT = 512
WIDTH = 512
FOCAL_LENGTH = 280


class VisibilityFilter:
  """Filters visibilities using semantic segmentation."""

  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
      print(
          "WARNING: No GPU available, using CPU instead for semantic"
          " segmentation."
      )

    self.model, self.preprocess, self.categories = (
        self.load_segmentation_model()
    )

  def load_segmentation_model(self) -> ...:
    """Loads FCN ResNet50 semantic segmentation model."""
    weights = torch_seg.FCN_ResNet50_Weights.DEFAULT
    assert (
        weights.url
        == "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth"
    )
    model = torch_seg.fcn_resnet50(weights=weights)
    model.eval()
    model.to(self.device)
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories

  def extract_masks(self, rgb_jpegs: list[bytes]) -> torch.Tensor:
    """Extracts semantic masks using a semantic segmentation model."""
    class_to_idx = {cls: idx for (idx, cls) in enumerate(self.categories)}
    rgb_ims = [
        torch.tensor(np.array(tf.io.decode_jpeg(x))).permute(2, 0, 1)
        for x in rgb_jpegs
    ]
    rgb_ims = torch.stack(rgb_ims, axis=0)
    batches = np.array_split(
        np.arange(rgb_ims.shape[0]), rgb_ims.shape[0] // 32
    )
    masks = []
    for batch_inds in batches:
      batch = self.preprocess(rgb_ims[batch_inds]).to(self.device)
      with torch.no_grad():
        prediction = self.model(batch)["out"].cpu()
      normalized_masks = prediction.softmax(dim=1)
      mask = normalized_masks[:, class_to_idx["person"]]
      masks.append(mask > 0.1)
    masks = torch.concatenate(masks, axis=0).cpu().numpy()
    return masks

  def filter_visibilities(
      self,
      trajectories: list[npt.NDArray],
      visibilities: list[npt.NDArray],
      rgb_jpegs: list[bytes],
  ):
    """Filters visibilities using semantic segmentation."""
    masks = self.extract_masks(rgb_jpegs)

    # Mark points on persons as occluded.
    px = trajectories[..., 0] * FOCAL_LENGTH / trajectories[..., 2] + (
        WIDTH / 2
    )
    py = trajectories[..., 1] * FOCAL_LENGTH / trajectories[..., 2] + (
        HEIGHT / 2
    )
    visib_filtered = []
    for t in range(px.shape[0]):
      row = np.floor(py[t]).astype(np.int32)
      col = np.floor(px[t]).astype(np.int32)
      visib = visibilities[t]
      semantic = np.array(Image.fromarray(masks[t]).resize((512, 512)))
      visib[visib] = ~semantic[row[visib], col[visib]]
      visib_filtered.append(visib)
    visibilities = np.stack(visib_filtered, axis=0)

    return visibilities


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
        selected_device_number, "skeleton" in sequence_path
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

  def get_queries_in_3d(
      self,
      queries_xyt: list[npt.NDArray],
      depth_ims: list[npt.NDArray],
      segmentation_ims: list[npt.NDArray],
      chunk_timestamps_ns: list[int],
  ) -> list[tuple[int, npt.NDArray]]:
    """Converts 2D queries to 3D queries in the object's coordinate frame."""
    queries_3d = []
    for x, y, query_time in queries_xyt:
      query_time = int(query_time)
      # Flip back 90 degs.
      x, y = y, HEIGHT - x
      row = int(np.round(y - 0.5))
      col = int(np.round(x - 0.5))

      query_3d_z = depth_ims[query_time][row, col] / 1000
      query_3d_x = (x - WIDTH / 2) * query_3d_z / FOCAL_LENGTH
      query_3d_y = (y - HEIGHT / 2) * query_3d_z / FOCAL_LENGTH

      query_3d = np.array([query_3d_x, query_3d_y, query_3d_z, 1.0])[..., None]

      # Get object id.
      select_timestamps_ns = chunk_timestamps_ns[query_time]
      target_obj_id = segmentation_ims[query_time][row, col]

      # Get object pose and bounding box.
      bbox3d_with_dt = (
          self.gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
              select_timestamps_ns
          )
      )
      if not bbox3d_with_dt.is_valid():
        continue
      if target_obj_id not in bbox3d_with_dt.data():
        continue
      bbox3d = bbox3d_with_dt.data()[target_obj_id]
      # Get the Aria pose.
      aria3dpose_with_dt = self.gt_provider.get_aria_3d_pose_by_timestamp_ns(
          select_timestamps_ns
      )
      if not aria3dpose_with_dt.is_valid():
        print("aria 3d pose is not available")
      aria3dpose = aria3dpose_with_dt.data()

      # Get 6DoF object pose with respect to the target camera.
      transform_cam_device = self.gt_provider.get_aria_transform_device_camera(
          self.stream_id
      ).inverse()
      transform_cam_scene = (
          transform_cam_device.to_matrix()
          @ aria3dpose.transform_scene_device.inverse().to_matrix()
      )
      transform_cam_obj = (
          transform_cam_scene @ bbox3d.transform_scene_object.to_matrix()
      )

      # Compute point coords in object frame.
      query_3d_obj = np.linalg.inv(transform_cam_obj) @ query_3d
      queries_3d.append((target_obj_id, query_3d_obj))
    return queries_3d

  def get_trajectories(
      self,
      queries_3d: list[npt.NDArray],
      depth_ims: list[npt.NDArray],
      chunk_timestamps_ns: list[int],
  ) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes trajectories and visibilities for all objects."""
    object_ids = np.array([q[0] for q in queries_3d])
    unique_object_ids = np.unique(object_ids)

    # Collect transforms to obj for all objs.
    transforms_cam_obj = dict()
    transform_cam_device = self.gt_provider.get_aria_transform_device_camera(
        self.stream_id
    ).inverse()
    for idx, select_timestamps_ns in enumerate(chunk_timestamps_ns):
      # Get the Aria pose.
      aria3dpose_with_dt = self.gt_provider.get_aria_3d_pose_by_timestamp_ns(
          select_timestamps_ns
      )
      if not aria3dpose_with_dt.is_valid():
        continue
      aria3dpose = aria3dpose_with_dt.data()
      transform_cam_scene = (
          transform_cam_device.to_matrix()
          @ aria3dpose.transform_scene_device.inverse().to_matrix()
      )
      for target_obj_id in unique_object_ids:
        bbox3d_with_dt = (
            self.gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
                select_timestamps_ns
            )
        )
        if not bbox3d_with_dt.is_valid():
          continue
        bbox3d = bbox3d_with_dt.data()[target_obj_id]
        transform_cam_obj = (
            transform_cam_scene @ bbox3d.transform_scene_object.to_matrix()
        )
        transforms_cam_obj[(idx, target_obj_id)] = transform_cam_obj

    queries_3d_obj = np.concatenate([q[1] for q in queries_3d], axis=-1).T[
        ..., None
    ]

    stacked_transforms_cam_obj = []
    for idx in range(len(chunk_timestamps_ns)):
      tnf_cam_obj = []
      for obj_id in object_ids:
        tnf_cam_obj.append(transforms_cam_obj[(idx, obj_id)])
      tnf_cam_obj = np.stack(tnf_cam_obj, axis=0)
      stacked_transforms_cam_obj.append(tnf_cam_obj)

    # Project to cam and compute visibility.
    trajectories = []
    visibilities = []
    for idx in range(len(chunk_timestamps_ns)):
      queries_3d_cam = (stacked_transforms_cam_obj[idx] @ queries_3d_obj)[
          :, :3, 0
      ]
      trajectories.append(queries_3d_cam)
      queries_depth = queries_3d_cam[:, 2]
      px = queries_3d_cam[:, 0] * FOCAL_LENGTH / queries_depth + WIDTH / 2
      py = queries_3d_cam[:, 1] * FOCAL_LENGTH / queries_depth + HEIGHT / 2
      in_frame = (
          (px > 0)
          * (py > 0)
          * (px < WIDTH)
          * (py < HEIGHT)
          * (queries_depth > 0)
      )
      visible = np.zeros_like(px).astype(bool)
      depth_at_query_in_frame = (
          interp_utils.interp_img(
              depth_ims[idx][..., None],
              np.stack((px[in_frame], py[in_frame]), axis=-1),
          ).squeeze()
          / 1000
      )
      visible_in_frame = (
          np.abs(queries_depth[in_frame] - depth_at_query_in_frame) < 0.02
      )
      visible[in_frame] = visible_in_frame
      visibilities.append(visible)

    visibilities = np.stack(visibilities, axis=0)
    trajectories = np.stack(trajectories, axis=0)

    # Flip 90deg.
    trajectories = trajectories[:, :, [1, 0, 2]]
    trajectories[:, :, 0] = -trajectories[:, :, 0]

    return trajectories, visibilities


def process_vid(
    input_adt_path: str,
    input_npz_path: str,
    output_npz_path: str,
    seq_name: str,
    chunks: list[int],
    visibility_filter: VisibilityFilter,
):
  """Processes multiple chunks of a single video."""
  sequence_path = glob.glob(os.path.join(input_adt_path, seq_name + "*"))[0]
  adt_processor = ADTVideoProcessor(sequence_path)

  for chunk_idx in tqdm.tqdm(chunks):
    track_fn = os.path.join(output_npz_path, f"{seq_name}_{chunk_idx}.npz")
    chunk_timestamps_ns = adt_processor.timestamps_ns[
        chunk_idx * N_FRAMES : (chunk_idx + 1) * N_FRAMES
    ]

    rgb_ims, depth_ims, segmentation_ims, chunk_timestamps_ns = (
        adt_processor.extract_image_data(chunk_timestamps_ns)
    )

    # Load query points.
    in_npz = np.load(
        os.path.join(input_npz_path, f"{seq_name}_{chunk_idx}.npz"),
        allow_pickle=True,
    )
    queries_xyt = in_npz["queries_xyt"]

    # Compute trajectories and visibilities from ADT source data.
    queries_3d = adt_processor.get_queries_in_3d(
        queries_xyt,
        depth_ims,
        segmentation_ims,
        chunk_timestamps_ns,
    )
    trajectories, visibilities = adt_processor.get_trajectories(
        queries_3d,
        depth_ims,
        chunk_timestamps_ns,
    )

    # Filter visibilities using semantic segmentation.
    rgb_ims = [np.array(Image.fromarray(im).rotate(-90)) for im in rgb_ims]
    rgb_jpegs = [np.array(tf.io.encode_jpeg(im)).item() for im in rgb_ims]
    visibilities = visibility_filter.filter_visibilities(
        trajectories,
        visibilities,
        rgb_jpegs,
    )

    # Verify trajectories to millimeter precision.
    trajectories_hash = hashlib.md5(
        np.round(trajectories * 1000).astype(np.int32).data.tobytes()
    ).hexdigest()
    assert trajectories_hash == in_npz["tracks_XYZ_hash"]

    # Verify visibilities to 1e-5 precision.
    assert np.abs(visibilities.mean() - in_npz["visibility_mean"]) < 1e-5

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

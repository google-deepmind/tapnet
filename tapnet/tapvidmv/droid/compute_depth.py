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

"""Stage 1: Stereo depth estimation and wrist-camera gripper refinement."""

import glob
import json
import os
import sys

from absl import app
import config as config_lib
import core.depth
import core.geometry
import core.io
import core.runner
import cv2
import h5py
import mediapy as media
from ml_collections import config_flags
import numpy as np
import torch
from tqdm import tqdm


def init_all_models(device):
  """Loads the S2M2 stereo model and SAM segmentation predictor."""
  vendor_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "third_party"
  )
  sys.path.append(os.path.join(vendor_dir, "s2m2", "src"))

  # pylint: disable=g-import-not-at-top
  from s2m2.core.utils.model_utils import load_model
  from segment_anything import SamPredictor
  from segment_anything import sam_model_registry
  # pylint: enable=g-import-not-at-top

  s2m2_model = load_model(
      os.path.join(vendor_dir, "s2m2", "weights"), "XL", device=device
  ).eval()
  sam_ckpt = os.path.join(
      vendor_dir, "segment_anything", "weights", "sam_vit_h_4b8939.pth"
  )
  sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)

  return s2m2_model, SamPredictor(sam)


def camera_matrix(cam):
  """Builds a 3x3 pinhole intrinsic matrix from ZED calibration parameters."""
  return np.array(
      [[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float32
  )


def init_episode(episode_id, root_path, id_to_path, serials_db):
  """Initializes the episode dictionary with paths and camera baselines."""
  cam_info = serials_db[episode_id]
  wrist_cam_id = cam_info["wrist_cam_serial"]

  return {
      "meta": {
          "episode_id": episode_id,
          "episode_path": os.path.join(root_path, id_to_path[episode_id]),
          "wrist_cam_id": wrist_cam_id,
      },
      "robot": {},
      "camera": {
          cam_id: {"baseline": 0.063 if cam_id == wrist_cam_id else 0.120}
          for cam_id in sorted(set(cam_info.values()))
      },
  }


def extract_svo_video(episode, min_frames, max_frames):
  """Decodes stereo RGB streams, intrinsics, and timestamps from SVO files."""
  from pyzed import sl  # pylint: disable=g-import-not-at-top

  episode_path = episode["meta"]["episode_path"]

  for cam_id in episode["camera"]:
    svo_files = glob.glob(
        os.path.join(episode_path, f"**/{cam_id}.svo"), recursive=True
    )
    if not svo_files:
      raise core.runner.SkipEpisode(f"{cam_id}: missing SVO in {episode_path}")
    svo_file = svo_files[0]

    zed, init_params = sl.Camera(), sl.InitParameters()
    init_params.set_from_svo_file(svo_file)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
      raise core.runner.SkipEpisode(f"{cam_id}: cannot open {svo_file} ({err})")

    n_svo_frames = zed.get_svo_number_of_frames() - 2
    if not min_frames <= n_svo_frames <= max_frames:
      zed.close()
      raise core.runner.SkipEpisode(
          f"{cam_id}: {n_svo_frames} frames outside [{min_frames},"
          f" {max_frames}]"
      )

    cam_info = zed.get_camera_information()
    configuration = cam_info.camera_configuration

    views = {"video_rgb": sl.VIEW.LEFT, "video_right": sl.VIEW.RIGHT}
    mats = {key: sl.Mat() for key in views}
    frames = {key: [] for key in views}
    timestamps = []

    for _ in tqdm(range(n_svo_frames + 2), desc=f"Decoding {cam_id}"):
      if zed.grab() != sl.ERROR_CODE.SUCCESS:
        continue

      timestamps.append(
          zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
      )
      for key, view in views.items():
        zed.retrieve_image(mats[key], view)
        frames[key].append(
            cv2.cvtColor(mats[key].get_data(), cv2.COLOR_BGRA2RGB)
        )

    zed.close()

    episode["camera"][cam_id].update(
        {key: np.stack(images) for key, images in frames.items()},
        K=camera_matrix(configuration.calibration_parameters.left_cam),
        timestamps=np.array(timestamps),
    )

  return episode


def parse_robot_kinematics(episode):
  """Loads joint states, end-effector poses, and wrist extrinsics from HDF5."""
  episode_path = episode["meta"]["episode_path"]

  with h5py.File(f"{episode_path}/trajectory.h5", "r") as f:
    ee_poses = f["observation/robot_state/cartesian_position"][:]
    joint_poses = f["observation/robot_state/joint_positions"][:]
    gripper_poses = f["observation/robot_state/gripper_position"][:]

    timestamps = f["observation/timestamp/robot_state/read_start"][:]

  with open(glob.glob(f"{episode_path}/metadata_*.json")[0]) as jf:
    wrist_ext = json.load(jf)["wrist_cam_extrinsics"]

  episode["robot"] = {
      "joint_positions": joint_poses.astype(np.float32),
      "gripper_positions": gripper_poses.astype(np.float32),
      "T_cam2ee": (
          np.linalg.inv(core.geometry.pose_from_euler(ee_poses[0]))
          @ core.geometry.pose_from_euler(wrist_ext)
      ),
      "T_ee2base": core.geometry.pose_from_euler(ee_poses),
      "timestamps": timestamps,
  }
  return episode


def align_temporal_streams(episode):
  """Truncates robot and camera streams to their common frame count."""
  robot_streams = [
      "joint_positions",
      "gripper_positions",
      "T_ee2base",
      "timestamps",
  ]
  camera_streams = ["video_rgb", "video_right", "timestamps"]

  streams = [(episode["robot"], key) for key in robot_streams]
  for cam_data in episode["camera"].values():
    for key in camera_streams:
      streams.append((cam_data, key))

  n_frames = min(len(owner[key]) for owner, key in streams)
  for owner, key in streams:
    owner[key] = owner[key][:n_frames]

  return episode


def export_depth(episode, export_root):
  """Writes stereo videos, depth maps, intrinsics, and robot state to disk."""
  wrist_cam_id = episode["meta"]["wrist_cam_id"]
  episode_dir = os.path.join(export_root, episode["meta"]["episode_id"])
  os.makedirs(episode_dir, exist_ok=True)

  for cam_id, data in episode["camera"].items():
    cam_dir = os.path.join(episode_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    for key, filename in [
        ("video_rgb", "video_left.mp4"),
        ("video_right", "video_right.mp4"),
    ]:
      media.write_video(os.path.join(cam_dir, filename), data[key], fps=10)

    np.savez_compressed(
        os.path.join(cam_dir, "raw_depth.npz"),
        depth=(data["raw_depth"] * 1000).astype(np.uint16),
    )

    np.savez(
        os.path.join(cam_dir, "calibration.npz"),
        K=data["K"],
        baseline=np.float32(data["baseline"]),
    )

    if cam_id == wrist_cam_id:
      np.savez_compressed(
          os.path.join(cam_dir, "gripper_mask.npz"), mask=data["sam_real_masks"]
      )
      for key, filename in [
          ("original_raw_depth", "original_raw_depth.npz"),
          ("empirical_gripper_depth", "gripper_depth.npz"),
      ]:
        np.savez_compressed(
            os.path.join(cam_dir, filename),
            depth=(data[key] * 1000).astype(np.uint16),
        )

  robot = episode["robot"]
  np.savez_compressed(
      os.path.join(episode_dir, "robot.npz"),
      wrist_cam_id=np.array(wrist_cam_id),
      joint_positions=robot["joint_positions"],
      gripper_positions=robot["gripper_positions"],
      T_ee2base=robot["T_ee2base"],
      T_cam2ee=robot["T_cam2ee"],
  )


def process_episode(
    episode_id,
    s2m2_model,
    sam_predictor,
    device,
    id_to_path,
    serials_db,
    config,
):
  """Runs Stage 1 depth extraction and gripper refinement for one episode."""
  episode = init_episode(episode_id, config.paths.raw, id_to_path, serials_db)
  episode = extract_svo_video(
      episode, config.depth.min_frames, config.depth.max_frames
  )
  episode = parse_robot_kinematics(episode)
  episode = align_temporal_streams(episode)
  episode = core.depth.compute_stereo_depth(
      episode, s2m2_model, device, config.depth.conf_thresh
  )
  episode = core.depth.refine_gripper_depth(
      episode, sam_predictor, config.depth
  )
  export_depth(episode, export_root=config.paths.depth)


def main(_):
  config = config_flag.value
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  s2m2_model, sam_predictor = init_all_models(device)
  serials_db, id_to_path, _, valid_ids = core.io.load_metadata(config)

  target = core.runner.select_episodes(valid_ids, config)
  done = {
      e
      for e in target
      if os.path.exists(os.path.join(config.paths.depth, e, "robot.npz"))
  }

  def run_one(episode_id):
    process_episode(
        episode_id,
        s2m2_model,
        sam_predictor,
        device,
        id_to_path,
        serials_db,
        config,
    )

  core.runner.run_episodes(target, run_one, done=done, stage="Stage 1")


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config_lib.__file__)
  app.run(main)

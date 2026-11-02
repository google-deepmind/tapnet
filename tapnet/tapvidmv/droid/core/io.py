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

"""Metadata and stage artifact I/O helpers for DROID episodes."""

import json
import os

import mediapy as media
import numpy as np


def load_metadata(config):
  """Downloads (if missing) and parses DROID camera and extrinsic metadata."""
  root_path = config.paths.meta
  os.makedirs(root_path, exist_ok=True)

  def fetch(name):
    os.system(f"wget -q -nc -P {root_path} {config.urls.meta}/{name}")
    with open(os.path.join(root_path, name), "r") as fh:
      return json.load(fh)

  serials_db = fetch("camera_serials.json")
  id_to_path = fetch("episode_id_to_path.json")
  extrinsics_db = fetch("cam2base_extrinsic_superset.json")

  valid_ids = []
  for episode_id in sorted(
      set(serials_db) & set(id_to_path) & set(extrinsics_db)
  ):
    cam_info = serials_db[episode_id]
    external_cams = set(cam_info.values()) - {cam_info["wrist_cam_serial"]}
    calibrated_cams = {
        cam_id
        for cam_id, entry in extrinsics_db[episode_id].items()
        if isinstance(entry, list)
    }
    if external_cams == calibrated_cams:
      valid_ids.append(episode_id)

  return serials_db, id_to_path, extrinsics_db, valid_ids


def load_depth_data(episode_id, depth_root, load_video=False):
  """Loads Stage 1 depth, calibration, kinematics, and optional RGB videos."""
  episode_dir = os.path.join(depth_root, episode_id)

  robot_data = np.load(os.path.join(episode_dir, "robot.npz"))
  wrist_cam_id = str(robot_data["wrist_cam_id"])
  robot_keys = ("joint_positions", "gripper_positions", "T_ee2base", "T_cam2ee")

  camera = {}
  for cam_id in sorted(
      d
      for d in os.listdir(episode_dir)
      if os.path.isdir(os.path.join(episode_dir, d))
  ):
    cam_dir = os.path.join(episode_dir, cam_id)
    calibration = np.load(os.path.join(cam_dir, "calibration.npz"))

    cam_data = {
        "K": calibration["K"].astype(np.float32),
        "baseline": float(calibration["baseline"]),
        "raw_depth": (
            np.load(os.path.join(cam_dir, "raw_depth.npz"))["depth"].astype(
                np.float32
            )
            / 1000.0
        ),
    }

    if load_video:
      cam_data["video_rgb"] = media.read_video(
          os.path.join(cam_dir, "video_left.mp4")
      )
      cam_data["video_right"] = media.read_video(
          os.path.join(cam_dir, "video_right.mp4")
      )

    if cam_id == wrist_cam_id:
      cam_data["sam_real_masks"] = np.load(
          os.path.join(cam_dir, "gripper_mask.npz")
      )["mask"]
      for filename, key in [
          ("original_raw_depth.npz", "original_raw_depth"),
          ("gripper_depth.npz", "empirical_gripper_depth"),
      ]:
        cam_data[key] = (
            np.load(os.path.join(cam_dir, filename))["depth"].astype(np.float32)
            / 1000.0
        )

    camera[cam_id] = cam_data

  return {
      "meta": {"episode_id": episode_id, "wrist_cam_id": wrist_cam_id},
      "robot": {key: robot_data[key].astype(np.float32) for key in robot_keys},
      "camera": camera,
  }


def load_extrinsics(episode, extrinsics_root):
  """Loads Stage 2 refined camera extrinsics for all cameras in an episode."""
  episode_id = episode["meta"]["episode_id"]
  episode_dir = os.path.join(extrinsics_root, episode_id)

  poses = {}
  for cam_id in episode["camera"]:
    with open(os.path.join(episode_dir, cam_id, "extrinsics.json"), "r") as f:
      payload = json.load(f)

    poses[cam_id] = {
        "T_cam2mount": np.array(payload["T_cam2mount"], dtype=np.float32),
        "T_cam2world": np.array(payload["T_cam2world"], dtype=np.float32),
    }

  return poses


def load_metrics(episode_id, extrinsics_root):
  episode_dir = os.path.join(extrinsics_root, episode_id)
  with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
    return json.load(f)


def load_track_data(episode_id, tracks_root):
  episode_dir = os.path.join(tracks_root, episode_id)
  with np.load(os.path.join(episode_dir, "tracks.npz")) as raw:
    return {
        "tracks_3d": raw["tracks_3d"].swapaxes(0, 1),
        "vis": raw["vis"].swapaxes(1, 2),
        "query_view": raw["query_view"],
        "query_frame": raw["query_frame"],
        "n_robot": int(raw["n_robot"]),
        "n_static": int(raw["n_static"]),
    }

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

"""Stage 2: Multi-camera and robot-arm extrinsic alignment optimization."""

# pylint: disable=invalid-name

import gc
import itertools
import json
import os

from absl import app
import config as config_lib
import core.geometry
import core.io
import core.physics
import core.pointcloud
import core.runner
from ml_collections import config_flags
import numpy as np
import torch
from torch import optim


def world_extrinsics(base, ee_poses, is_wrist):
  """Computes per-frame camera-to-world poses from mount pose and EE trajectory."""
  if is_wrist:
    return ee_poses @ base
  return np.tile(base, (len(ee_poses), 1, 1))


def init_camera_states(episode, extrinsics_db):
  """Initializes camera mount and world poses from metadata and kinematics."""
  print("  Initializing camera 3D states...")
  wrist_cam_id = episode["meta"]["wrist_cam_id"]
  robot_data = episode["robot"]
  episode_extrinsics = extrinsics_db[episode["meta"]["episode_id"]]

  poses = {}
  for cam_id in episode["camera"]:
    is_wrist = cam_id == wrist_cam_id
    T_cam2mount = (
        robot_data["T_cam2ee"]
        if is_wrist
        else core.geometry.pose_from_euler(episode_extrinsics[cam_id])
    )
    if not is_wrist:
      print(
          f"    Loaded pre-calibrated extrinsics for camera [{cam_id}] from"
          " metadata."
      )
    poses[cam_id] = {
        "T_cam2mount": T_cam2mount,
        "T_cam2world": world_extrinsics(
            T_cam2mount, robot_data["T_ee2base"], is_wrist
        ),
    }

  return poses


def alignment_loss(pose, data, wrist_cam_id, pairs, config):
  """Computes cross-camera Chamfer and robot disparity alignment losses."""
  chamfer, _ = core.pointcloud.chamfer_overlap_px(
      data["env"],
      data["ee_poses"],
      pose,
      wrist_cam_id,
      pairs,
      data["K"],
      data["baseline"],
      data["disparity"],
      config.extrinsics.chamfer_px_radius,
  )
  disparity = [
      core.pointcloud.disparity_loss_batched(
          points,
          pose[cam_id],
          data["K"][cam_id],
          data["depth_batch"][cam_id],
          data["valid_batch"][cam_id],
          data["baseline"][cam_id],
          config.extrinsics.disparity_truncation,
      )
      for cam_id, points in data["robot_points"].items()
  ]
  return sum(chamfer.values()), sum(disparity)


def optimize_poses(base, data, wrist_cam_id, pairs, label, device, config):
  """Optimizes SE(3) pose deltas for all cameras jointly with Adam."""
  n_steps = config.extrinsics.n_steps
  delta = {
      cam_id: torch.zeros(6, requires_grad=True, device=device)
      for cam_id in base
  }

  def posed():
    return {
        cam_id: base[cam_id] @ core.geometry.pose_from_axis_angle(delta[cam_id])
        for cam_id in base
    }

  optimizer = optim.Adam(list(delta.values()), lr=config.extrinsics.lr)
  schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
  for step in range(n_steps):
    optimizer.zero_grad()
    chamfer, disparity = alignment_loss(
        posed(), data, wrist_cam_id, pairs, config
    )
    (chamfer + config.extrinsics.robot_weight * disparity).backward()
    optimizer.step()
    schedule.step()

    if step % 100 == 0 or step == n_steps - 1:
      shifts = ", ".join(
          f"{label[cam_id]}:"
          f" {torch.norm(delta[cam_id][:3]).item() * 1000:.2f}mm"
          for cam_id in base
      )
      print(
          f"    Step {step:03d} | chamfer_px: {chamfer.item():.4f} | disparity:"
          f" {disparity.item():.4f} | Shift: {shifts}"
      )

  with torch.no_grad():
    return posed()


def joint_alignment(episode, prev_poses, render_pool, device, config):
  """Prepares scene and robot point clouds and refines camera extrinsics."""
  n_points = config.extrinsics.n_points
  print("\nJoint optimization...")

  wrist_cam_id = episode["meta"]["wrist_cam_id"]
  cam_ids = [c for c in episode["camera"] if c != wrist_cam_id] + [wrist_cam_id]
  pairs = list(itertools.combinations(cam_ids, 2))
  fixed_cam_ids = [c for c in cam_ids if c != wrist_cam_id]
  label = {c: str(i + 1) for i, c in enumerate(fixed_cam_ids)} | {
      wrist_cam_id: "W"
  }
  T_ee2base = episode["robot"]["T_ee2base"]

  data = {"baseline": {c: episode["camera"][c]["baseline"] for c in cam_ids}}
  data["env"], data["ee_poses"], frames = core.pointcloud.scene_clouds(
      episode, device, n_points
  )
  data["disparity"] = core.pointcloud.disparity_maps(episode, frames, device)

  renders = render_pool.render_cameras(episode, prev_poses)
  data["robot_points"], data["depth_batch"], data["K"] = (
      core.pointcloud.robot_clouds(
          episode, prev_poses, renders, device, n_points
      )
  )
  data["valid_batch"] = {
      c: (d > 0.0).float() for c, d in data["depth_batch"].items()
  }

  base = {
      c: torch.tensor(
          prev_poses[c]["T_cam2mount"], dtype=torch.float32, device=device
      )
      for c in cam_ids
  }
  final = optimize_poses(base, data, wrist_cam_id, pairs, label, device, config)
  return {
      cam_id: {
          "T_cam2mount": final[cam_id].cpu().numpy(),
          "T_cam2world": world_extrinsics(
              final[cam_id].cpu().numpy(), T_ee2base, cam_id == wrist_cam_id
          ),
      }
      for cam_id in cam_ids
  }


def export_extrinsics(episode, poses, metrics, export_root):
  """Saves refined camera extrinsics and alignment quality metrics as JSON."""
  episode_id = episode["meta"]["episode_id"]
  episode_dir = os.path.join(export_root, episode_id)
  fname = "extrinsics.json"

  for cam_id, state in poses.items():
    cam_dir = os.path.join(episode_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    payload = {
        "T_cam2mount": state["T_cam2mount"].tolist(),
        "T_cam2world": state["T_cam2world"].tolist(),
    }

    out_path = os.path.join(cam_dir, fname)
    with open(out_path, "w") as f:
      json.dump(payload, f, indent=2)

  payload = {
      k: None if isinstance(v, float) and not np.isfinite(v) else v
      for k, v in metrics.items()
  }
  with open(os.path.join(episode_dir, "metrics.json"), "w") as f:
    json.dump(payload, f, indent=2)

  print(f"  Extrinsics and metrics saved to {episode_dir}")


@torch.no_grad()
def evaluate_extrinsics(episode, poses, render_pool, device, config):
  """Evaluates cross-camera Chamfer overlap and robot depth alignment metrics."""
  wrist_cam_id = episode["meta"]["wrist_cam_id"]
  cam_ids = sorted(episode["camera"])
  pairs = list(itertools.combinations(cam_ids, 2))
  base = {
      c: torch.as_tensor(
          poses[c]["T_cam2mount"], dtype=torch.float32, device=device
      )
      for c in cam_ids
  }

  n_points, max_depth = config.extrinsics.n_points, config.extrinsics.max_depth
  renders = render_pool.render_cameras(episode, poses)
  robot_points, depth_batch, K = core.pointcloud.robot_clouds(
      episode, poses, renders, device, n_points
  )

  near_env, near_ee, _ = core.pointcloud.scene_clouds(
      episode, device, n_points, max_depth
  )
  chamfer, overlap = core.pointcloud.chamfer_overlap(
      near_env,
      near_ee,
      base,
      wrist_cam_id,
      pairs,
      config.extrinsics.chamfer_match_radius,
  )

  env, ee_poses, frames = core.pointcloud.scene_clouds(
      episode, device, n_points
  )
  chamfer_px, overlap_px = core.pointcloud.chamfer_overlap_px(
      env,
      ee_poses,
      base,
      wrist_cam_id,
      pairs,
      K,
      {c: episode["camera"][c]["baseline"] for c in cam_ids},
      core.pointcloud.disparity_maps(episode, frames, device),
      config.extrinsics.chamfer_px_radius,
  )
  robot = {
      c: core.pointcloud.depth_loss_batched(
          robot_points[c], base[c], K[c], depth_batch[c], max_depth
      )
      for c in cam_ids
  }

  return (
      {f"chamfer_{a}_{b}": chamfer[a, b].item() for a, b in pairs}
      | {f"overlap_{a}_{b}": overlap[a, b].item() * 100 for a, b in pairs}
      | {f"chamfer_px_{a}_{b}": chamfer_px[a, b].item() for a, b in pairs}
      | {f"overlap_px_{a}_{b}": overlap_px[a, b].item() * 100 for a, b in pairs}
      | {f"robot_loss_{cam_id}": robot[cam_id].item() for cam_id in cam_ids}
  )


def episode_metrics(episode, poses, render_pool, device, config):
  """Builds the full summary and alignment metric dictionary for an episode."""
  episode_id = episode["meta"]["episode_id"]
  site, scene, _ = episode_id.split("+")
  return {
      "episode_id": episode_id,
      "site": site,
      "scene": scene,
      "n_cameras": len(episode["camera"]),
      "n_frames": len(episode["robot"]["joint_positions"]),
      "wrist_cam_id": episode["meta"]["wrist_cam_id"],
  } | evaluate_extrinsics(episode, poses, render_pool, device, config)


def process_episode(episode_id, render_pool, extrinsics_db, device, config):
  """Runs Stage 2 extrinsic optimization and evaluation for one episode."""
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = init_camera_states(episode, extrinsics_db)
  poses = joint_alignment(episode, poses, render_pool, device, config)
  metrics = episode_metrics(episode, poses, render_pool, device, config)

  export_extrinsics(
      episode, poses, metrics, export_root=config.paths.extrinsics
  )

  gc.collect()
  torch.cuda.empty_cache()


def main(_):
  config = config_flag.value
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  _, _, extrinsics_db, _ = core.io.load_metadata(config)
  render_pool = core.physics.RenderPool(
      core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu),
      config.render.workers,
  )

  available = core.runner.finished(config.paths.depth, "robot.npz")
  target = core.runner.select_episodes(available, config)
  done = {
      e
      for e in target
      if os.path.exists(
          os.path.join(config.paths.extrinsics, e, "metrics.json")
      )
  }

  def run_one(episode_id):
    process_episode(episode_id, render_pool, extrinsics_db, device, config)

  core.runner.run_episodes(target, run_one, done=done, stage="Stage 2")


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config_lib.__file__)
  app.run(main)

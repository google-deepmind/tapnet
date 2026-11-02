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

"""Point cloud sampling and differentiable Chamfer/disparity alignment losses."""

# pylint: disable=invalid-name

import core.geometry
import numpy as np
import torch
import torch.nn.functional as F


def sample_camera_points(mask, z, K, n_points):
  v, u = np.where(mask)
  if len(u) < n_points:
    return None

  idx = np.random.choice(len(u), n_points, replace=False)
  v, u = v[idx], u[idx]
  return core.geometry.unproject_camera_frame(u, v, z[v, u], K)


def extract_robot_clouds(
    K, rendered, T_cam2mount, device, depth_batch, n_points
):
  """Samples mount-frame 3D robot surface clouds from rendered depth maps."""
  cache_X, kept = [], []
  for t in range(len(rendered)):
    points_cam = sample_camera_points(rendered[t] > 0, rendered[t], K, n_points)
    if points_cam is None:
      continue

    cache_X.append(
        torch.tensor(
            (T_cam2mount @ points_cam)[:3, :].T,
            dtype=torch.float32,
            device=device,
        )
    )
    kept.append(t)

  return torch.stack(cache_X), depth_batch[kept]


def robot_clouds(episode, poses, renders, device, n_points):
  """Builds batched robot surface point clouds and observed depth tensors."""
  robot_points, depth_batch, K = {}, {}, {}
  for cam_id, cam_data in episode["camera"].items():
    observed = torch.as_tensor(
        cam_data["raw_depth"], dtype=torch.float32, device=device
    ).unsqueeze(1)
    robot_points[cam_id], depth_batch[cam_id] = extract_robot_clouds(
        cam_data["K"],
        renders[cam_id],
        poses[cam_id]["T_cam2mount"],
        device,
        observed,
        n_points,
    )
    K[cam_id] = torch.as_tensor(
        cam_data["K"], dtype=torch.float32, device=device
    )

  return robot_points, depth_batch, K


def scene_clouds(episode, device, n_points, max_depth=np.inf):
  """Samples camera-frame 3D scene point clouds across valid episode frames."""
  cameras = episode["camera"]
  T_ee2base = episode["robot"]["T_ee2base"]

  cache = {cam_id: [] for cam_id in cameras}
  cache_ee, frames = [], []
  for t in range(len(episode["robot"]["joint_positions"])):
    frame = {}
    for cam_id, cam_data in cameras.items():
      depth = cam_data["raw_depth"][t]
      frame[cam_id] = sample_camera_points(
          (depth > 0.0) & (depth < max_depth), depth, cam_data["K"], n_points
      )
    if all(points is not None for points in frame.values()):
      for cam_id, points in frame.items():
        cache[cam_id].append(
            torch.as_tensor(points, dtype=torch.float32, device=device)
        )
      cache_ee.append(
          torch.as_tensor(T_ee2base[t], dtype=torch.float32, device=device)
      )
      frames.append(t)

  return (
      {cam_id: torch.stack(clouds) for cam_id, clouds in cache.items()},
      torch.stack(cache_ee),
      np.array(frames),
  )


def disparity_maps(episode, frames, device):
  maps = {}
  for cam_id, cam_data in episode["camera"].items():
    depth = torch.as_tensor(
        cam_data["raw_depth"][frames], dtype=torch.float32, device=device
    )
    maps[cam_id] = torch.where(
        depth > 0,
        cam_data["K"][0, 0] * cam_data["baseline"] / depth.clamp(min=1e-6),
        0.0,
    )
  return maps


def batched_chamfer_distance(p1, p2, match_radius):
  """Computes truncated symmetric Chamfer distance and overlap ratio."""
  dist = torch.cdist(p1, p2)
  near_12 = dist.min(dim=2)[0]
  near_21 = dist.min(dim=1)[0]

  valid_12 = near_12 < match_radius
  valid_21 = near_21 < match_radius
  loss = (near_12 * valid_12).sum() / valid_12.sum().clamp(min=1)
  loss = loss + (near_21 * valid_21).sum() / valid_21.sum().clamp(min=1)

  overlap = (valid_12.sum() + valid_21.sum()) / (
      p1.shape[0] * (p1.shape[1] + p2.shape[1])
  )
  return loss, overlap


def sample_at_projection(points, T_cam2world, K, images):
  """Projects 3D points into image space and bilinearly samples image values."""
  _, _, height, width = images.shape

  points_cam = (points - T_cam2world[:3, 3]) @ T_cam2world[:3, :3]
  z_pred = points_cam[..., 2]

  u = K[0, 0] * points_cam[..., 0] / z_pred + K[0, 2]
  v = K[1, 1] * points_cam[..., 1] / z_pred + K[1, 2]

  grid = torch.stack(
      [(u / (width - 1)) * 2 - 1, (v / (height - 1)) * 2 - 1], dim=-1
  ).unsqueeze(1)
  sampled = (
      F.grid_sample(
          images,
          grid,
          mode="bilinear",
          padding_mode="border",
          align_corners=True,
      )
      .squeeze(1)
      .squeeze(1)
  )

  uv_in_frame = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
  return sampled, z_pred, uv_in_frame & (z_pred > 0.0)


def depth_loss_batched(points, T_cam2world, K, depth_batch, max_depth):
  z_obs, z_pred, in_frame = sample_at_projection(
      points, T_cam2world, K, depth_batch
  )
  valid = in_frame & (z_pred < max_depth) & (z_obs > 0.0) & (z_obs < max_depth)
  return torch.abs(z_obs[valid] - z_pred[valid]).mean()


def disparity_loss_batched(
    points, T_cam2world, K, depth_batch, valid_batch, baseline, truncation
):
  """Computes truncated stereo disparity residual between robot and depth."""
  z_obs, z_pred, in_frame = sample_at_projection(
      points, T_cam2world, K, depth_batch
  )
  measured, _, _ = sample_at_projection(points, T_cam2world, K, valid_batch)
  residual = (
      K[0, 0]
      * baseline
      * torch.abs(1.0 / z_obs.clamp(min=1e-6) - 1.0 / z_pred.clamp(min=1e-6))
  )
  residual = torch.where(
      in_frame,
      residual.clamp(max=truncation),
      torch.full_like(residual, truncation),
  )
  counted = ~in_frame | (measured > 0.999)
  return residual[counted].mean()


def world_clouds(env, ee_poses, pose, wrist_cam_id):
  to_world, world = {}, {}
  for cam_id, cloud in env.items():
    to_world[cam_id] = (
        ee_poses @ pose[cam_id] if cam_id == wrist_cam_id else pose[cam_id]
    )
    world[cam_id] = (to_world[cam_id] @ cloud)[:, :3, :].transpose(1, 2)
  return to_world, world


def chamfer_overlap(env, ee_poses, pose, wrist_cam_id, pairs, match_radius):
  _, world = world_clouds(env, ee_poses, pose, wrist_cam_id)
  chamfer, overlap = {}, {}
  for a, b in pairs:
    chamfer[a, b], overlap[a, b] = batched_chamfer_distance(
        world[a], world[b], match_radius
    )

  return chamfer, overlap


def sparse_to_dense(points_world, T_cam2world, K, baseline, disparity, radius):
  """Computes pixel-disparity nearest-neighbor distance against dense maps."""
  _, height, width = disparity.shape
  points_cam = (
      points_world - T_cam2world[..., :3, 3].unsqueeze(-2)
  ) @ T_cam2world[..., :3, :3]
  z = points_cam[..., 2].clamp(min=1e-6)
  u = K[0, 0] * points_cam[..., 0] / z + K[0, 2]
  v = K[1, 1] * points_cam[..., 1] / z + K[1, 2]
  d = K[0, 0] * baseline / z

  steps = torch.arange(
      -int(np.ceil(radius)),
      int(np.ceil(radius)) + 1,
      device=points_world.device,
  )
  du, dv = torch.meshgrid(steps, steps, indexing="xy")
  pu = u.detach().round().unsqueeze(-1) + du.flatten()
  pv = v.detach().round().unsqueeze(-1) + dv.flatten()
  inside = (
      (pu >= 0)
      & (pu < width)
      & (pv >= 0)
      & (pv < height)
      & (points_cam[..., 2] > 0).unsqueeze(-1)
  )
  index = (pv.clamp(0, height - 1) * width + pu.clamp(0, width - 1)).long()
  observed = torch.gather(disparity.flatten(1), 1, index.flatten(1)).view(
      index.shape
  )

  gap = (
      (u.unsqueeze(-1) - pu) ** 2
      + (v.unsqueeze(-1) - pv) ** 2
      + (d.unsqueeze(-1) - observed) ** 2
  )
  gap = torch.where(inside & (observed > 0), gap, torch.inf)
  nearest = torch.sqrt(gap.min(dim=-1).values + 1e-12)
  return nearest, nearest < radius


def chamfer_overlap_px(
    env, ee_poses, pose, wrist_cam_id, pairs, K, baseline, disparity, radius
):
  """Computes pairwise pixel-disparity Chamfer loss and overlap across views."""
  to_world, world = world_clouds(env, ee_poses, pose, wrist_cam_id)
  chamfer, overlap = {}, {}
  for a, b in pairs:
    loss, hits, total = 0.0, 0, 0
    for source, target in ((b, a), (a, b)):
      nearest, matched = sparse_to_dense(
          world[source],
          to_world[target],
          K[target],
          baseline[target],
          disparity[target],
          radius,
      )
      loss = loss + torch.where(
          matched, nearest, 0.0
      ).sum() / matched.sum().clamp(min=1)
      hits, total = hits + matched.sum(), total + matched.numel()
    chamfer[a, b], overlap[a, b] = loss, hits / total

  return chamfer, overlap

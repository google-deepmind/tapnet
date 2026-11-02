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

"""Camera projection, unprojection, and SE(3) pose utilities."""

# pylint: disable=invalid-name

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def decode_disparity(disparity, fx, baseline):
  z = np.zeros_like(disparity)
  valid_mask = disparity > 0
  z[valid_mask] = (fx * baseline) / disparity[valid_mask]
  z[~np.isfinite(z)] = 0.0
  return z


def unproject_camera_frame(u, v, z, K):
  u, v = u.astype(z.dtype), v.astype(z.dtype)
  return np.stack(
      [
          (u - K[0, 2]) * z / K[0, 0],
          (v - K[1, 2]) * z / K[1, 1],
          z,
          np.ones_like(z),
      ],
      axis=0,
  )


def unproject_pixels(u, v, z, K, T_cam2world):
  return (T_cam2world @ unproject_camera_frame(u, v, z, K))[:3, :].T


def project_points(points_world, K, T_cam2world):
  T_world2cam = np.linalg.inv(T_cam2world)
  points_cam = (
      points_world @ np.swapaxes(T_world2cam[..., :3, :3], -1, -2)
      + T_world2cam[..., None, :3, 3]
  )
  z = points_cam[..., 2]
  front = z > 0
  u, v = np.zeros_like(z), np.zeros_like(z)
  u[front] = points_cam[..., 0][front] / z[front] * K[0, 0] + K[0, 2]
  v[front] = points_cam[..., 1][front] / z[front] * K[1, 1] + K[1, 2]
  return u, v, z


def farthest_points(points, n_points, seeds=None):
  chosen = []
  if seeds is None or np.asarray(seeds).size == 0:
    nearest = np.sum((points - points.mean(axis=0)) ** 2, axis=1)
  else:
    nearest = np.sum((points[:, None] - seeds[None]) ** 2, axis=-1).min(axis=1)
  for _ in range(min(n_points, len(points))):
    pick = int(np.argmax(nearest))
    chosen.append(pick)
    nearest = np.minimum(nearest, np.sum((points - points[pick]) ** 2, axis=1))
    nearest[pick] = -np.inf
  return np.array(chosen, dtype=int)


def in_frame(u, v, width, height):
  return (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)


def sample_depth(depth, u, v, z):
  height, width = depth.shape
  ui = np.clip(np.round(u).astype(int), 0, width - 1)
  vi = np.clip(np.round(v).astype(int), 0, height - 1)
  return np.where(
      in_frame(u, v, width, height) & (z > 0), depth[vi, ui], np.nan
  )


def unproject_depth_torch(depth, img_rgb, K, T_cam2world, device):
  depth = torch.as_tensor(depth, dtype=torch.float32, device=device)
  v, u = torch.nonzero(depth > 0, as_tuple=True)
  z = depth[v, u]
  K = torch.as_tensor(K, dtype=torch.float32, device=device)
  points_cam = torch.stack(
      [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dim=1
  )
  T = torch.as_tensor(T_cam2world, dtype=torch.float32, device=device)
  points_world = points_cam @ T[:3, :3].T + T[:3, 3]
  return points_world, torch.as_tensor(img_rgb, device=device)[v, u]


def pose_from_euler(vec_6d):
  vec_6d = np.asarray(vec_6d)
  transform = np.tile(np.eye(4, dtype=np.float32), (*vec_6d.shape[:-1], 1, 1))
  transform[..., :3, :3] = R.from_euler("xyz", vec_6d[..., 3:]).as_matrix()
  transform[..., :3, 3] = vec_6d[..., :3]
  return transform


def pose_from_axis_angle(delta):
  x, y, z = delta[3:]
  zero = torch.zeros_like(x)
  skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).view(3, 3)
  pose = torch.eye(4, device=delta.device)
  pose[:3, :3] = torch.linalg.matrix_exp(skew)
  pose[:3, 3] = delta[:3]
  return pose

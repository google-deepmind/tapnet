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

"""Preprocessing and coordinate-transform utilities for TAPNext++ inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_frame(
    frame_bgr: np.ndarray,
    device: torch.device,
    model_size: int = 256,
) -> torch.Tensor:
  """Convert a BGR uint8 frame to a model-ready tensor.

  Resize and normalise on the GPU to minimise CPU work for high-resolution
  camera or video inputs.

  Args:
      frame_bgr: [H, W, 3] uint8 BGR frame (OpenCV layout).
      device: Target device.
      model_size: Square resolution expected by the model.

  Returns:
      Float32 tensor of shape [1, 1, model_size, model_size, 3] in [-1, 1].
  """
  frame_rgb = frame_bgr[..., ::-1].copy()  # BGR → RGB
  t = torch.from_numpy(frame_rgb).to(
      device, non_blocking=True
  )  # [H, W, 3] uint8
  t = t.permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
  t = F.interpolate(
      t, size=(model_size, model_size), mode='bilinear', align_corners=False
  )
  t = t.div_(127.5).sub_(1.0)
  return t.permute(0, 2, 3, 1).unsqueeze(0)  # [1, 1, S, S, 3]


def display_to_model(
    points_xy: np.ndarray,
    disp_h: int,
    disp_w: int,
    model_size: int = 256,
) -> np.ndarray:
  """Scale [N, 2] [x, y] points from display pixels to model-space pixels.

  Model space is always ``model_size × model_size`` regardless of frame
  aspect ratio.

  Args:
    points_xy: [N, 2] float32 array of [x, y] coordinates in display pixels.
    disp_h: Height of the display frame.
    disp_w: Width of the display frame.
    model_size: Square resolution expected by the model.

  Returns:
    [N, 2] float32 array of [x, y] coordinates in model space.
  """
  scale = np.array([model_size / disp_w, model_size / disp_h], dtype=np.float32)
  return (points_xy * scale).astype(np.float32)


def model_to_display(
    points_xy: np.ndarray,
    disp_h: int,
    disp_w: int,
    model_size: int = 256,
) -> np.ndarray:
  """Scale [N, 2] [x, y] points from model-space pixels to display pixels.

  Args:
    points_xy: [N, 2] float32 array of [x, y] coordinates in model space.
    disp_h: Height of the display frame.
    disp_w: Width of the display frame.
    model_size: Square resolution expected by the model.

  Returns:
    [N, 2] float32 array of [x, y] coordinates in display pixels.
  """
  scale = np.array([disp_w / model_size, disp_h / model_size], dtype=np.float32)
  return (points_xy * scale).astype(np.float32)


def make_query_tensor(
    model_pts_xy: np.ndarray,
    device: torch.device,
    query_timestep: int = 0,
) -> torch.Tensor:
  """Build a [1, Q, 3] query tensor in TAPNext [t, y, x] format.

  Args:
      model_pts_xy: [Q, 2] float32 array of [x, y] coordinates in model space.
      device: Target device.
      query_timestep: Frame index at which the query points are defined (0 =
        current / first frame).

  Returns:
      [1, Q, 3] float32 tensor with layout [t, y, x].
  """
  q = len(model_pts_xy)
  query = np.zeros((q, 3), dtype=np.float32)
  query[:, 0] = query_timestep
  query[:, 1] = model_pts_xy[:, 1]  # y
  query[:, 2] = model_pts_xy[:, 0]  # x
  return torch.from_numpy(query).to(device).unsqueeze(0)  # [1, Q, 3]

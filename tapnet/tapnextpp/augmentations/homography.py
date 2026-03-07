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

"""Homography augmentation for TAP-Next long context training."""

import cv2
import numpy as np


def _reflect(val, min_b, max_b):
  while val < min_b or val > max_b:
    if val < min_b:
      val = min_b + (min_b - val)
    if val > max_b:
      val = max_b - (val - max_b)
  return val


def get_sinusoid_pert(t, n_low, n_high, amps, freqs, phases):
  pert = 0
  for i in range(n_low):
    pert += amps[0][i] * (
        np.sin(t * freqs[0][i] + phases[0][i]) - np.sin(phases[0][i])
    )
  for i in range(n_high):
    pert += amps[1][i] * (
        np.sin(t * freqs[1][i] + phases[1][i]) - np.sin(phases[1][i])
    )
  return pert


class HomographyAugmentation:
  """Homography augmentation for video frames and trajectories.

  This augmentation mimics camera motion by applying projective transform
  to each frame of a video sequence. Trajectories are adjusted
  accordingly.
  """

  def __init__(
      self,
      *,
      p: float = 0.8,
      strength: float = 1.0,
      border_mode: str = 'constant',
      debug: bool = False,
  ) -> None:
    self.p = p
    self.strength = strength
    self.border_mode = border_mode
    if border_mode == 'constant':
      self.cv_border_mode = cv2.BORDER_CONSTANT
    elif border_mode == 'replicate':
      self.cv_border_mode = cv2.BORDER_REPLICATE
    else:
      raise ValueError(f'Unknown border mode: {border_mode}')
    self.debug = debug

  def __call__(self, data):
    if np.random.rand() > self.p and not self.debug:
      return data

    video = data['rgb/encoded']
    trajs = data['trajs_2d'].copy()

    s, h, w, _ = video.shape

    n_low = 3
    n_high = 3

    # 8 perturbations for 4 corners (x,y for each)
    # amplitudes are relative to image size later
    pert_params = []
    for _ in range(8):
      if self.debug:
        low_freq_amp = np.full(n_low, 0.05) * self.strength
        high_freq_amp = np.full(n_high, 0.02) * self.strength
      else:
        low_freq_amp = np.random.uniform(0, 0.05, n_low) * self.strength
        high_freq_amp = np.random.uniform(0, 0.02, n_high) * self.strength
      low_freq_freq = np.random.uniform(1, 4, n_low) * np.pi
      low_freq_phase = np.random.uniform(0, 2 * np.pi, n_low)
      high_freq_freq = np.random.uniform(8, 16, n_high) * np.pi
      high_freq_phase = np.random.uniform(0, 2 * np.pi, n_high)
      pert_params.append((
          (low_freq_amp, high_freq_amp),
          (low_freq_freq, high_freq_freq),
          (low_freq_phase, high_freq_phase),
          n_low,
          n_high,
      ))

    pts_src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )

    augmented_video_list = []

    for i in range(s):
      img = video[i]
      t = i / s if s > 1 else 0

      perts_flat = []
      signs = [1, 1, -1, 1, -1, -1, 1, -1]
      for j, params in enumerate(pert_params):
        amps, freqs, phases, n_low_p, n_high_p = params
        pert = get_sinusoid_pert(t, n_low_p, n_high_p, amps, freqs, phases)
        pert_coord = np.abs(pert) * w if j % 2 == 0 else np.abs(pert) * h
        perts_flat.append(signs[j] * pert_coord)

      perts = np.array(perts_flat).reshape(4, 2)
      pts_dst = pts_src + perts.astype(np.float32)

      # Mirror points into corner regions to prevent flipping
      w_margin = w * 0.3
      h_margin = h * 0.3
      pts_dst[0, 0] = _reflect(pts_dst[0, 0], 0, w_margin)  # TL_x
      pts_dst[0, 1] = _reflect(pts_dst[0, 1], 0, h_margin)  # TL_y
      pts_dst[1, 0] = _reflect(pts_dst[1, 0], w - 1 - w_margin, w - 1)  # TR_x
      pts_dst[1, 1] = _reflect(pts_dst[1, 1], 0, h_margin)  # TR_y
      pts_dst[2, 0] = _reflect(pts_dst[2, 0], w - 1 - w_margin, w - 1)  # BR_x
      pts_dst[2, 1] = _reflect(pts_dst[2, 1], h - 1 - h_margin, h - 1)  # BR_y
      pts_dst[3, 0] = _reflect(pts_dst[3, 0], 0, w_margin)  # BL_x
      pts_dst[3, 1] = _reflect(pts_dst[3, 1], h - 1 - h_margin, h - 1)  # BL_y

      m = cv2.getPerspectiveTransform(pts_src, pts_dst)

      aug_img = cv2.warpPerspective(
          img, m, (w, h), borderMode=self.cv_border_mode
      )
      augmented_video_list.append(aug_img)

      points = trajs[i].reshape(-1, 1, 2)
      if points.shape[0] > 0:
        transformed_points = cv2.perspectiveTransform(points, m).reshape(-1, 2)
        trajs[i, :, 0] = transformed_points[:, 0]
        trajs[i, :, 1] = transformed_points[:, 1]

    data['rgb/encoded'] = np.stack(augmented_video_list)
    data['trajs_2d'] = trajs
    return data

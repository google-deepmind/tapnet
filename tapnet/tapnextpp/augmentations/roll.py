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

"""Roll augmentation for TAP-Next long context training."""

import cv2
import numpy as np


class RollAugmentation:
  """Roll and rotation augmentation for video frames and trajectories.

  This augmentation mimics camera roll and jitter by applying sinusoidal shifts
  and rotations to each frame of a video sequence. Trajectories are adjusted
  accordingly.
  """

  def __init__(self, rotate=True, p=0.8, strength=1.0):
    self.rotate = rotate
    self.p = p
    self.strength = strength

  def __call__(self, data):
    if np.random.rand() > self.p:
      return data

    video = data['rgb/encoded']
    trajs = data['trajs_2d'].copy()

    s, h, w, _ = video.shape

    padding = 0
    if self.rotate:
      padding = int(np.ceil((np.sqrt(h**2 + w**2) - min(h, w)) / 2.0))
      padded_video_list = []
      for i in range(s):
        padded_video_list.append(
            np.pad(
                video[i],
                ((padding, padding), (padding, padding), (0, 0)),
                mode='constant',
            )
        )
      video_padded = np.stack(padded_video_list)
      trajs[..., 0] += padding
      trajs[..., 1] += padding
    else:
      video_padded = video.copy()

    _, padded_h, padded_w, _ = video_padded.shape

    n_low = 5
    n_high = 5

    low_freq_x_amp = np.random.uniform(0, 30, n_low) * self.strength
    low_freq_x_freq = np.random.uniform(1, 4, n_low) * np.pi
    low_freq_x_phase = np.random.uniform(0, 2 * np.pi, n_low)
    high_freq_x_amp = np.random.uniform(0, 10, n_high) * self.strength
    high_freq_x_freq = np.random.uniform(8, 16, n_high) * np.pi
    high_freq_x_phase = np.random.uniform(0, 2 * np.pi, n_high)

    low_freq_y_amp = np.random.uniform(0, 20, n_low) * self.strength
    low_freq_y_freq = np.random.uniform(1, 4, n_low) * np.pi
    low_freq_y_phase = np.random.uniform(0, 2 * np.pi, n_low)
    high_freq_y_amp = np.random.uniform(0, 7, n_high) * self.strength
    high_freq_y_freq = np.random.uniform(8, 16, n_high) * np.pi
    high_freq_y_phase = np.random.uniform(0, 2 * np.pi, n_high)

    low_freq_rot_amp = None
    low_freq_rot_freq = None
    low_freq_rot_phase = None
    high_freq_rot_amp = None
    high_freq_rot_freq = None
    high_freq_rot_phase = None

    if self.rotate:
      low_freq_rot_amp = np.random.uniform(0, 10, n_low) * self.strength
      low_freq_rot_freq = np.random.uniform(1, 4, n_low) * np.pi
      low_freq_rot_phase = np.random.uniform(0, 2 * np.pi, n_low)
      high_freq_rot_amp = np.random.uniform(0, 5, n_high) * self.strength
      high_freq_rot_freq = np.random.uniform(8, 16, n_high) * np.pi
      high_freq_rot_phase = np.random.uniform(0, 2 * np.pi, n_high)

    augmented_video_list = []

    # --- 1. Vectorized calculation of t, shift, and angle for all frames ---
    # Generate a time array of shape (S,)
    t_array = np.arange(s) / s if s > 1 else np.zeros(s)

    shift_x_arr = np.zeros(s)
    shift_y_arr = np.zeros(s)
    angle_arr = np.zeros(s)

    # Accumulate low-frequency components
    for j in range(n_low):
      shift_x_arr += low_freq_x_amp[j] * (
          np.sin(t_array * low_freq_x_freq[j] + low_freq_x_phase[j])
          - np.sin(low_freq_x_phase[j])
      )
      shift_y_arr += low_freq_y_amp[j] * (
          np.sin(t_array * low_freq_y_freq[j] + low_freq_y_phase[j])
          - np.sin(low_freq_y_phase[j])
      )
      if self.rotate:
        angle_arr += low_freq_rot_amp[j] * (
            np.sin(t_array * low_freq_rot_freq[j] + low_freq_rot_phase[j])
            - np.sin(low_freq_rot_phase[j])
        )

    # Accumulate high-frequency components
    for j in range(n_high):
      shift_x_arr += high_freq_x_amp[j] * (
          np.sin(t_array * high_freq_x_freq[j] + high_freq_x_phase[j])
          - np.sin(high_freq_x_phase[j])
      )
      shift_y_arr += high_freq_y_amp[j] * (
          np.sin(t_array * high_freq_y_freq[j] + high_freq_y_phase[j])
          - np.sin(high_freq_y_phase[j])
      )
      if self.rotate:
        angle_arr += high_freq_rot_amp[j] * (
            np.sin(t_array * high_freq_rot_freq[j] + high_freq_rot_phase[j])
            - np.sin(high_freq_rot_phase[j])
        )

    # Round shifts to the nearest integer and cast to int
    shift_x_arr = np.round(shift_x_arr).astype(int)
    shift_y_arr = np.round(shift_y_arr).astype(int)

    # --- 2. Iterate through each frame to apply transformations ---
    for i in range(s):
      img = video_padded[i]
      shift_x = shift_x_arr[i]
      shift_y = shift_y_arr[i]

      # Apply translation (shift)
      rolled_img = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))

      # Synchronize trajectory point coordinates
      x_coords = trajs[i, :, 0] + shift_x
      y_coords = trajs[i, :, 1] + shift_y

      trajs[i, :, 0] = x_coords % padded_w
      trajs[i, :, 1] = y_coords % padded_h

      if self.rotate:
        h, w = rolled_img.shape[:2]
        center = (w // 2, h // 2)
        angle = angle_arr[i]

        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img_padded = cv2.warpAffine(
            rolled_img, m, (w, h), borderMode=cv2.BORDER_CONSTANT
        )

        points = trajs[i].reshape(-1, 1, 2)
        transformed_points = cv2.transform(points, m).reshape(-1, 2)

        x_coords_rot = transformed_points[:, 0]
        y_coords_rot = transformed_points[:, 1]

        trajs[i, :, 0] = x_coords_rot
        trajs[i, :, 1] = y_coords_rot

        augmented_video_list.append(aug_img_padded)
      else:
        augmented_video_list.append(rolled_img)

    augmented_video = np.stack(augmented_video_list)

    if self.rotate:
      data['rgb/encoded'] = augmented_video[
          :, padding : padding + h, padding : padding + w, :
      ]
      trajs[..., 0] -= padding
      trajs[..., 1] -= padding
    else:
      data['rgb/encoded'] = augmented_video

    data['trajs_2d'] = trajs
    return data

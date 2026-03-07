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

"""Visualizes roll augmentation for TAP-Next long context training."""

import argparse
import cv2
import imageio
import numpy as np
from roll import RollAugmentation


def create_dummy_frame(frame_idx, total_frames, height=256, width=256):
  """Creates a dummy frame with a moving circle."""
  img = np.full((height, width, 3), 255, dtype=np.uint8)  # White background
  center_x, center_y = width // 2, height // 2

  # moving circle
  circle_radius = 20
  path_radius = 50
  angle = 16 * np.pi * frame_idx / total_frames
  circle_x = center_x + int(path_radius * np.cos(angle))
  circle_y = center_y + int(path_radius * np.sin(angle))

  cv2.circle(img, (circle_x, circle_y), circle_radius, (0, 0, 0), -1)
  tl_x, tl_y = 10, 20
  br_x, br_y = width - 50, height - 20
  cv2.putText(
      img, 'TL', (tl_x, tl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
  )
  cv2.putText(
      img,
      'BR',
      (br_x, br_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 0, 255),
      1,
  )
  return img


def visualize_roll(output_path):
  """Generates and saves frames demonstrating roll augmentation."""
  gif_frames = []
  h, w = 256, 256
  s = 128  # num frames

  video = []
  for i in range(s):
    video.append(create_dummy_frame(i, s, height=h, width=w))
  video = np.stack(video)

  data = {
      'rgb/encoded': video,
      'trajs_2d': np.zeros((s, 1, 2)),
  }

  aug = RollAugmentation(p=1.1, strength=1.0)  # force augmentation
  augmented_data = aug(data)

  for i in range(s):
    gif_frames.append(
        cv2.cvtColor(augmented_data['rgb/encoded'][i], cv2.COLOR_BGR2RGB)
    )

  imageio.mimsave(output_path, gif_frames, fps=30)
  print(f'Saved GIF to {output_path}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output_path',
      default='/tmp/roll_viz.gif',
      help='Output path for visualization GIF.',
  )
  args = parser.parse_args()
  visualize_roll(args.output_path)

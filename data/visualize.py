# Copyright 2022 DeepMind Technologies Limited
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

"""Visualize frames of a random video of the given dataset."""

import colorsys
import math
import pickle
import random
from typing import List, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import cv2
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', None, 'Path to the pickle file.', required=True)
flags.DEFINE_string(
    'output_path', None, 'Path to the output mp4 video.', required=True)


# Generate random colormaps for visualizing different points.
def _get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0., 360., 360. / num_colors):
    hue = i / 360.
    lightness = (50 + np.random.rand() * 10) / 100.
    saturation = (90 + np.random.rand() * 10) / 100.
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
  random.shuffle(colors)
  return colors


_COLORS = _get_colors(num_colors=70)


def paint_point_track(
    frames: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
) -> np.ndarray:
  """Returns frames with painted points."""
  painted_frames = []
  for idx in range(len(frames)):
    frame = frames[idx]
    for track in range(len(points)):
      x, y = points[track, idx]
      occ = occluded[track, idx]
      if not occ:
        height, width, _ = frame.shape
        x = x * width
        y = y * height
        color = _COLORS[track]

        x_floor, x_ceil = math.floor(x), math.ceil(x)
        y_floor, y_ceil = math.floor(y), math.ceil(y)

        new_frame = np.zeros_like(frame, dtype=np.float64)
        new_frame = cv2.circle(
            frame, (int(x_floor), int(y_floor)),
            radius=3,
            color=color,
            thickness=-1) * (x_ceil - x) * (
                y_ceil - y)
        new_frame += cv2.circle(
            frame, (int(x_floor), int(y_ceil)),
            radius=3,
            color=color,
            thickness=-1) * (x_ceil - x) * (
                y - y_floor)
        new_frame += cv2.circle(
            frame, (int(x_ceil), int(y_floor)),
            radius=3,
            color=color,
            thickness=-1) * (x - x_floor) * (
                y_ceil - y)
        new_frame += cv2.circle(
            frame, (int(x_ceil), int(y_ceil)),
            radius=3,
            color=color,
            thickness=-1) * (x - x_floor) * (
                y - y_floor)
        new_frame = np.array(new_frame, dtype=np.uint8)
        frame = new_frame
    painted_frames.append(frame)

  return np.array(painted_frames)


def write_video(frames: np.ndarray, output_path: str) -> None:
  _, height, width, _ = frames.shape
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  video = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

  for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

  cv2.destroyAllWindows()
  video.release()


def main(argv: Sequence[str]) -> None:
  del argv

  input_path = FLAGS.input_path
  logging.info('Loading data in "%s". This may take some time.', input_path)
  with open(input_path, 'rb') as f:
    data = pickle.load(f)
    if isinstance(data, dict):
      data = list(data.values())

  idx = random.randint(0, len(data) - 1)
  video = data[idx]

  frames = video['video']

  if isinstance(frames[0], bytes):
    # Tapnet is stored and JPEG bytes rather than `np.ndarray`s.
    def decode(frame):
      frame = cv2.imdecode(
          np.frombuffer(frame, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)
      return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames = [decode(frame) for frame in frames]

  painted_frames = paint_point_track(frames, video['points'], video['occluded'])

  write_video(painted_frames, FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)

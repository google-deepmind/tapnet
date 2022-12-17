<<<<<<< HEAD
# Copyright 2022 DeepMind Technologies Limited.
=======
# Copyright 2022 DeepMind Technologies Limited
>>>>>>> cb3ef0ce00547001f108aa7094a4d0f2071e4079
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
<<<<<<< HEAD
#     http://www.apache.org/licenses/LICENSE-2.0
=======
#    http://www.apache.org/licenses/LICENSE-2.0
>>>>>>> cb3ef0ce00547001f108aa7094a4d0f2071e4079
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
<<<<<<< HEAD
=======
# ==============================================================================
>>>>>>> cb3ef0ce00547001f108aa7094a4d0f2071e4079

"""Visualize frames of a random video of the given dataset."""

import collections
import colorsys
import csv
import os
import random
from typing import List, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import mediapy as media
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_csv_path', None, 'Path to the csv file.', required=True
)
flags.DEFINE_string(
    'input_video_dir', None, 'Path to the videos', required=True
)
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


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
) -> np.ndarray:
  """Converts a sequence of points to color code video.

  Args:
    frames: [num_frames, height, width, 3], np.uint8, [0, 255]
    point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
    visibles: [num_points, num_frames], bool

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
  num_points, num_frames = point_tracks.shape[0:2]
  height, width = frames.shape[1:3]
  colormap = _get_colors(num_points)
  dot_size_as_fraction_of_min_edge = 0.015
  radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
  diam = radius * 2 + 1
  quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
  quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
  icon = (quadratic_y + quadratic_x) - (radius ** 2) / 2.
  sharpness = .15
  icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
  icon = 1 - icon[:, :, np.newaxis]
  icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
  icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
  icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
  icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

  video = frames.copy()
  for t in range(num_frames):
    # Pad so that points that extend outside the image frame don't crash us
    image = np.pad(
        video[t],
        [
            (radius + 1, radius + 1),
            (radius + 1, radius + 1),
            (0, 0),
        ],
    )
    for i in range(num_points):
      # The icon is centered at the center of a pixel, but the input coordinates
      # are raster coordinates.  Therefore, to render a point at (1,1) (which
      # lies on the corner between four pixels), we need 1/4 of the icon placed
      # centered on the 0'th row, 0'th column, etc.  We need to subtract
      # 0.5 to make the fractional position come out right.
      x, y = point_tracks[i, t, :] + 0.5
      x = min(max(x, 0.0), width)
      y = min(max(y, 0.0), height)

      if visibles[i, t]:
        x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x2, y2 = x1 + 1, y1 + 1

        # bilinear interpolation
        patch = (
            icon1 * (x2 - x) * (y2 - y) +
            icon2 * (x2 - x) * (y - y1) +
            icon3 * (x - x1) * (y2 - y) +
            icon4 * (x - x1) * (y - y1))
        x_ub = x1 + 2 * radius + 2
        y_ub = y1 + 2 * radius + 2
        image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
            y1:y_ub, x1:x_ub, :
        ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

      # Remove the pad
      video[t] = image[radius + 1:-radius - 1,
                       radius + 1:-radius - 1].astype(np.uint8)
  return video


def main(argv: Sequence[str]) -> None:
  del argv

  logging.info('Loading data from %s. This takes time.', FLAGS.input_csv_path)
  point_tracks_all = collections.defaultdict(list)
  with open(FLAGS.input_csv_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      youtube_id = row[0]
      point_tracks = np.array(row[3:]).reshape(-1, 3)
      point_tracks_all[youtube_id].append(point_tracks)

  video_id = random.choice(list(point_tracks_all.keys()))
  point_tracks = point_tracks_all[video_id]
  video_path = os.path.join(FLAGS.input_video_dir, video_id + '.mp4')
  frames = media.read_video(video_path)

  point_tracks = np.stack(point_tracks_all[video_id], axis=0)
  point_tracks, occluded = point_tracks[..., 0:2], point_tracks[..., 2]
  point_tracks = point_tracks.astype(np.float32)
  occluded = occluded.astype(bool)

  # The stored point tracks coordinates are normalized to [0, 1).
  # So we multiple by the width and height of the video [0, width/height).
  height, width = frames.shape[1:3]
  point_tracks *= np.array([width, height], dtype=np.float32)
  painted_frames = paint_point_track(frames, point_tracks, ~occluded)

  media.write_video(FLAGS.output_path, painted_frames, fps=25)
  logging.info('Examplar point visualization saved to %s', FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)

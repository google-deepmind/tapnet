# Copyright 2023 DeepMind Technologies Limited
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

"""Visualization utility functions."""

import colorsys
import random
from typing import List, Optional, Sequence, Tuple

from absl import logging
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np


# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
  """Converts a sequence of points to color code video.

  Args:
    frames: [num_frames, height, width, 3], np.uint8, [0, 255]
    point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
    visibles: [num_points, num_frames], bool
    colormap: colormap for points, each point has a different RGB color.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
  num_points, num_frames = point_tracks.shape[0:2]
  if colormap is None:
    colormap = get_colors(num_colors=num_points)
  height, width = frames.shape[1:3]
  dot_size_as_fraction_of_min_edge = 0.015
  radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
  diam = radius * 2 + 1
  quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
  quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
  icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
  sharpness = 0.15
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
            icon1 * (x2 - x) * (y2 - y)
            + icon2 * (x2 - x) * (y - y1)
            + icon3 * (x - x1) * (y2 - y)
            + icon4 * (x - x1) * (y - y1)
        )
        x_ub = x1 + 2 * radius + 2
        y_ub = y1 + 2 * radius + 2
        image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
            y1:y_ub, x1:x_ub, :
        ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

      # Remove the pad
      video[t] = image[
          radius + 1 : -radius - 1, radius + 1 : -radius - 1
      ].astype(np.uint8)
  return video


def plot_tracks_v2(
    rgb: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
    gt_points: Optional[np.ndarray] = None,
    gt_occluded: Optional[np.ndarray] = None,
    trackgroup: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Plot tracks with matplotlib."""
  disp = []
  cmap = plt.cm.hsv  # pytype: disable=module-attr

  z_list = (
      np.arange(points.shape[0]) if trackgroup is None else np.array(trackgroup)
  )

  # random permutation of the colors so nearby points in the list can get
  # different colors
  z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
  colors = cmap(z_list / (np.max(z_list) + 1))
  figure_dpi = 64

  figs = []
  for i in range(rgb.shape[0]):
    fig = plt.figure(
        figsize=(256 / figure_dpi, 256 / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w',
    )
    figs.append(fig)
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(rgb[i] / 255.0)
    colalpha = np.concatenate(
        [colors[:, :-1], 1 - occluded[:, i : i + 1]], axis=1
    )
    plt.scatter(points[:, i, 0], points[:, i, 1], s=9, c=colalpha)
    occ2 = occluded[:, i : i + 1]
    if gt_occluded is not None:
      occ2 *= 1 - gt_occluded[:, i : i + 1]
    colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

    if gt_points is not None:
      colalpha = np.concatenate(
          [colors[:, :-1], 1 - gt_occluded[:, i : i + 1]], axis=1
      )
      plt.scatter(
          gt_points[:, i, 0], gt_points[:, i, 1], s=15, c=colalpha, marker='D'
      )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3
    )
    disp.append(np.copy(img))

  for fig in figs:
    plt.close(fig)
  return np.stack(disp, axis=0)


def plot_tracks_v3(
    rgb: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
    gt_points: np.ndarray,
    gt_occluded: np.ndarray,
    trackgroup: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Plot tracks in a 2x2 grid."""
  if trackgroup is None:
    trackgroup = np.arange(points.shape[0])
  else:
    trackgroup = np.array(trackgroup)

  utg = np.unique(trackgroup)
  chunks = np.array_split(utg, 4)
  plots = []
  for ch in chunks:
    valid = np.any(trackgroup[:, np.newaxis] == ch[np.newaxis, :], axis=1)

    new_trackgroup = np.argmax(
        trackgroup[valid][:, np.newaxis] == ch[np.newaxis, :], axis=1
    )
    plots.append(
        plot_tracks_v2(
            rgb,
            points[valid],
            occluded[valid],
            None if gt_points is None else gt_points[valid],
            None if gt_points is None else gt_occluded[valid],
            new_trackgroup,
        )
    )
  p1 = np.concatenate(plots[0:2], axis=2)
  p2 = np.concatenate(plots[2:4], axis=2)
  return np.concatenate([p1, p2], axis=1)


def write_visualization(
    video: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
    visualization_path: Sequence[str],
    gt_points: Optional[np.ndarray] = None,
    gt_occluded: Optional[np.ndarray] = None,
    trackgroup: Optional[np.ndarray] = None,
):
  """Write a visualization."""
  for i in range(video.shape[0]):
    logging.info('rendering...')

    video_frames = plot_tracks_v2(
        video[i],
        points[i],
        occluded[i],
        gt_points[i] if gt_points is not None else None,
        gt_occluded[i] if gt_occluded is not None else None,
        trackgroup[i] if trackgroup is not None else None,
    )

    logging.info('writing...')
    with media.VideoWriter(
        visualization_path[i],
        shape=video_frames.shape[-3:-1],
        fps=5,
        codec='h264',
        bps=600000,
    ) as video_writer:
      for j in range(video_frames.shape[0]):
        fr = video_frames[j]
        video_writer.add_image(fr.astype(np.uint8))

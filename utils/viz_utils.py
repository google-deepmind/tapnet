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
import jax
import jax.numpy as jnp
import matplotlib
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
    point_size: int = 20,
) -> np.ndarray:
  """Plot tracks with matplotlib.

  This function also supports plotting ground truth tracks alongside
  predictions, and allows you to specify tracks that should be plotted
  with the same color (trackgroup).  Note that points which are out of
  bounds will be clipped to be within bounds; mark them as occluded if
  you don't want them to appear.

  Args:
    rgb: frames of shape [num_frames, height, width, 3].  Each frame is passed
      directly to plt.imshow.
    points: tracks, of shape [num_points, num_frames, 2], np.float32. [0, width
      / height]
    occluded: [num_points, num_frames], bool, True if the point is occluded.
    gt_points: Optional, ground truth tracks to be plotted with diamonds, same
      shape/dtype as points
    gt_occluded: Optional, ground truth occlusion values to be plotted with
      diamonds, same shape/dtype as occluded.
    trackgroup: Optional, shape [num_points], int: grouping labels for the
      plotted points.  Points with the same integer label will be plotted with
      the same color.  Useful for clustering applications.
    point_size: int, the size of the plotted points, passed as the 's' parameter
      to matplotlib.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
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
        figsize=(rgb.shape[2] / figure_dpi, rgb.shape[1] / figure_dpi),
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
    points = np.maximum(points, 0.0)
    points = np.minimum(points, [rgb.shape[2], rgb.shape[1]])
    plt.scatter(points[:, i, 0], points[:, i, 1], s=point_size, c=colalpha)
    occ2 = occluded[:, i : i + 1]
    if gt_occluded is not None:
      occ2 *= 1 - gt_occluded[:, i : i + 1]

    if gt_points is not None:
      colalpha = np.concatenate(
          [colors[:, :-1], 1 - gt_occluded[:, i : i + 1]], axis=1
      )
      plt.scatter(
          gt_points[:, i, 0],
          gt_points[:, i, 1],
          s=point_size + 6,
          c=colalpha,
          marker='D',
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


def estimate_homography(targ_pts, src_pts, mask=None):
  """Estimate a homography between two sets of points."""
  # Standard homography estimation from SVD.  See e.g.
  # https://www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/
  # lecture_4_3-estimating-homographies-from-feature-correspondences.pdf
  # for reference.
  if mask is None:
    mask = jnp.ones_like(targ_pts[..., 0])
  targ_x = targ_pts[..., 0]

  targ_y = targ_pts[..., 1]
  src_x = src_pts[..., 0]
  src_y = src_pts[..., 1]
  one = jnp.ones_like(targ_x)
  zero = jnp.zeros_like(targ_x)
  a1 = (
      jnp.stack(
          [
              src_x,
              src_y,
              one,
              zero,
              zero,
              zero,
              -targ_x * src_x,
              -targ_x * src_y,
              -targ_x,
          ],
          axis=-1,
      )
      * mask[:, jnp.newaxis]
  )
  a2 = (
      jnp.stack(
          [
              zero,
              zero,
              zero,
              src_x,
              src_y,
              one,
              -targ_y * src_x,
              -targ_y * src_y,
              -targ_y,
          ],
          axis=-1,
      )
      * mask[:, jnp.newaxis]
  )
  a = jnp.concatenate([a1, a2], axis=-2)
  if a.shape[0] <= 8:
    u, s, v = jnp.linalg.svd(a, full_matrices=True)  # pylint: disable=unused-variable
  else:
    u, s, v = jnp.linalg.svd(a, full_matrices=False)  # pylint: disable=unused-variable
  h = np.reshape(v[..., -1, :], (3, 3))
  return h


def compute_inliers(
    homog, thresh, targ_pts=None, src_pts=None, src_pts_homog=None
):
  """Compute inliers and errors."""
  if src_pts_homog is None:
    src_pts_homog = jnp.transpose(
        jnp.concatenate([src_pts, src_pts[:, 0:1] * 0 + 1], axis=-1)
    )
  tformed = jnp.transpose(jnp.matmul(homog, src_pts_homog))
  tformed = tformed[..., :-1] / (
      jnp.maximum(1e-12, jnp.abs(tformed[..., -1:]))
      * jnp.sign(tformed[..., -1:])
  )
  err = jnp.sum(jnp.square(targ_pts - tformed), axis=-1)
  new_inliers = err < thresh * thresh
  return new_inliers, err, tformed


def ransac_homography(targ_pts, src_pts, vis, thresh=4.0, targ_inlier_frac=0.5):
  """Run RANSAC."""
  targ_pts_choice, src_pts_choice = [], []
  probs = vis / jnp.sum(vis)
  perm = jax.vmap(
      lambda x: jax.random.choice(
          jax.random.PRNGKey(x), targ_pts.shape[0], [4], replace=False, p=probs
      )
  )(jnp.arange(targ_pts.shape[0], dtype=jnp.int32))
  targ_pts_choice = jnp.take_along_axis(
      targ_pts[:, jnp.newaxis], perm[:, :, jnp.newaxis], axis=0
  )
  src_pts_choice = jnp.take_along_axis(
      src_pts[:, jnp.newaxis], perm[:, :, jnp.newaxis], axis=0
  )
  src_pts_homog = jnp.transpose(
      jnp.concatenate([src_pts, src_pts[:, 0:1] * 0 + 1], axis=-1)
  )

  compute_inliers2 = lambda x: compute_inliers(
      x, thresh, targ_pts=targ_pts, src_pts_homog=src_pts_homog
  )[0]

  def loop_body(arg):
    it, inliers, old_homog = arg
    homog = estimate_homography(targ_pts_choice[it], src_pts_choice[it])
    new_inliers = compute_inliers2(homog)
    new_inliers = jnp.sum(jnp.array(new_inliers, jnp.int32))
    homog = jax.lax.cond(
        new_inliers > inliers, lambda: homog, lambda: old_homog
    )

    return (it + 1, jnp.maximum(inliers, new_inliers), homog)

  def should_continue(arg):
    it, inliers, _ = arg
    # first term guarantees we terminate before we run out of points
    # second term decays slowly from the target inlier fraction
    threshold = jnp.minimum(
        jnp.array(1 - (it + 1) / src_pts_choice.shape[0], jnp.float32),
        targ_inlier_frac * (0.99 ** jnp.array(it, jnp.float32)),
    )
    threshold = threshold * jnp.array(src_pts_choice.shape[0], jnp.float32)
    return jnp.array(inliers, jnp.float32) < threshold

  init = (0, 0, jnp.zeros([3, 3]))
  _, _, homog = jax.lax.while_loop(should_continue, loop_body, init)
  inliers = compute_inliers2(homog)
  final_homog = estimate_homography(
      targ_pts, src_pts, jnp.array(inliers, jnp.float32)
  )

  return final_homog, inliers


def maybe_ransac_homography(*arg, thresh=4.0, targ_inlier_frac=0.5):
  """Run RANSAC if there's enough points."""
  targ_pts_all, targ_occ, src_pts_all, src_occ = arg
  vis = jnp.logical_and(jnp.logical_not(targ_occ), jnp.logical_not(src_occ))
  if np.sum(vis) > 4:
    final_homog, _ = ransac_homography(
        targ_pts_all,
        src_pts_all,
        vis,
        thresh,
        targ_inlier_frac=targ_inlier_frac,
    )
  else:
    final_homog = jnp.eye(3)
  inliers, err, tformed = compute_inliers(
      final_homog, thresh, targ_pts=targ_pts_all, src_pts=src_pts_all
  )

  return final_homog, inliers, tformed, err


def compute_canonical_points(
    all_tformed, occ, err, inner_thresh, outer_thresh, required_inlier_frac
):
  """Compute canonical points."""
  definite_outliers = jnp.logical_or(occ, err > outer_thresh)
  maybe_inliers = jnp.logical_and(jnp.logical_not(occ), err < inner_thresh)
  sum_inliers = jnp.sum(maybe_inliers, axis=0)
  sum_vis = jnp.sum(jnp.logical_not(occ), axis=0)
  frac_inliers = sum_inliers / jnp.maximum(1.0, sum_vis)
  canonical_invalid = frac_inliers < required_inlier_frac
  canonical_pts = jnp.einsum(
      'tnc,tn->nc', all_tformed, np.logical_not(definite_outliers)
  ) / np.maximum(
      1.0, jnp.sum(np.logical_not(definite_outliers), axis=0)[:, jnp.newaxis]
  )
  # re-initialize invalid canonical points with random un-occluded values.
  # Note this will default to 0 for points that have no un-occluded values.
  vis = 1 - occ
  random_choice = np.floor(
      np.random.random([vis.shape[1]]) * jnp.sum(vis, axis=0)
  )
  ids = jnp.cumsum(vis, axis=0) * vis - 1 * occ
  idx = ids == random_choice[jnp.newaxis, :]
  idx = np.sum(
      idx * jnp.arange(vis.shape[0], dtype=jnp.int32)[:, jnp.newaxis], axis=0
  )[jnp.newaxis, :, jnp.newaxis]
  random_pts = np.take_along_axis(all_tformed, idx, axis=0)[0]
  canonical_pts = (
      canonical_invalid[:, jnp.newaxis] * random_pts
      + (1 - canonical_invalid[:, jnp.newaxis]) * canonical_pts
  )
  return canonical_pts, canonical_invalid


def get_homographies_wrt_frame(
    pts,
    occ,
    image_dimensions,
    reference_frame=None,
    thresh=0.07,
    outlier_point_threshold=0.95,
    targ_inlier_frac=0.7,
    num_refinement_passes=2,
):
  """Compute a set of homographies between each frame and a 'canonical frame'.

  The canonical frame is the normalized coordinates for reference_frame (i.e.
  the homography computed for reference_frame will be a diagonal matrix).

  The underlying algorithm begins by running ransac for every frame,
  starting with frames after the reference_frame (in ascending order) followed
  by previous frames (in descending order). Over time, the canonical positions
  are averages of point locations, and are used to determine outlier points.

  This function does not assume any order to the input points.  However, it
  assumes the background motion can be captured with a homography: that is,
  it only works if the camera center does not move (the camera can pan) or
  if the background is planar.  It also assumes that points on the camera plane
  of reference_frame are not visible in other frames (as these would have
  coordinates at infinity in the reference_frame).

  Args:
    pts: Points array, float32, of shape [num_points, num_frames, 2] in x,y
      order in raster coordinates.
    occ: Array of occlusion values, where 1 is occluded and 0 is not, of shape
      [num_points, num_frames].
    image_dimensions: 2-element list of [width, height] of the original image.
      For numerical stability, points are internally rescaled to the range [0,
      1].
    reference_frame: compute transformations with respect to this frame.
    thresh: outlier threshold, specified in the normalized coordinates (i.e.
      specified as a fraction of the width/height).  We consider a point to be
      an outlier if less than outlier_point_threshold of its visible points are
      at a distance larget than thresh.  When computing canonical points, we
      also drop any frames that are further than 2*thresh from the canonical
      points.
    outlier_point_threshold: see docs for thresh.  Require this many points to
      be within thresh to use this point for homography estimation.
    targ_inlier_frac: when running ransac, terminate if this fraction of points
      are inliers.  Note that when running ransac, this threshold will decay
      exponentially to ensure termination even if there aren't enough inliers.
      However, if you know that only a small number of points are inliers, you
      may be able to speed up the algorithm by lowering this value.
    num_refinement_passes: After initialization, we refine the homographies
      using the canonical points estimated globally.  Use this many passes for
      every frame.

  Returns:
    homogs: [num_frames, 3, 3] float tensor such that
      inv(homogs[i]) @ homogs[j] is a matrix that will map background points
      from frame j to frame i.
    err: Float array of shape [num_points, num_frames] where each entry is the
      distance between that point and the estimated canonical point in the
      canonical frame.  Useful for determining if a point is an outlier or not.
    canonical_pts:
      Float array of shape [num_points, 2] of estimated locations in the
      canonical frame.  Useful for sorting the points before displaying them.
  """
  # Due to legacy reasons, all the underlying functions have the frames on the
  # first axis and the different points on the second.  This is the opposite
  # of most of the functions in this codebase.
  pts = np.transpose(pts, (1, 0, 2)) / np.array(image_dimensions)
  occ = np.transpose(occ)
  outer_thresh = thresh * 2.0
  if reference_frame is None:
    reference_frame = pts.shape[0] // 2
  canonical_pts = pts[reference_frame]
  canonical_invalid = occ[reference_frame]
  all_tformed_pts = np.zeros_like(pts)
  all_tformed_invalid = np.ones_like(occ)
  all_err = np.zeros(occ.shape)
  all_tformed_pts[reference_frame] = canonical_pts
  all_tformed_invalid[reference_frame] = canonical_invalid
  res_homog = [None] * pts.shape[0]
  res_homog[reference_frame] = jnp.eye(3)

  after = list(range(reference_frame + 1, pts.shape[0]))
  before = list(range(reference_frame - 1, -1, -1))
  for i in after + before:
    print(f'Initial RANSAC frame {i}...')
    res, _, tformed, err = maybe_ransac_homography(
        canonical_pts,
        canonical_invalid,
        pts[i],
        occ[i],
        thresh=thresh,
        targ_inlier_frac=targ_inlier_frac,
    )
    all_tformed_pts[i] = tformed
    all_tformed_invalid[i] = occ[i]
    all_err[i] = err
    res_homog[i] = res
    canonical_pts, canonical_invalid = compute_canonical_points(
        all_tformed_pts,
        all_tformed_invalid,
        err,
        thresh,
        outer_thresh,
        outlier_point_threshold,
    )
  for j in range(num_refinement_passes):
    for fr in [reference_frame,] + after + before:
      print(f'Refinement pass {j} frame {fr}...')
      _, err, _ = compute_inliers(
          res_homog[fr], thresh, canonical_pts, pts[fr]
      )
      invalid = jnp.logical_or(canonical_invalid, err > thresh * thresh)
      invalid = jnp.logical_or(occ[fr], invalid)

      homog = estimate_homography(
          canonical_pts,
          pts[fr],
          jnp.array(jnp.logical_not(invalid), jnp.float32),
      )
      if fr == reference_frame and j != num_refinement_passes - 1:
        # The scale of the reference frame is pinned to the original
        # points.  Therefore, when we solve for the reference frame's
        # homography, we actually apply the inverse homography to all of the
        # other frames.  Not we don't update error estimates; they'll be
        # slightly inaccurate until they're updated again.
        #
        # But don't do that on the last iteration because it causes an obvious
        # jump (i.e. allow the solution to collapse a little at the end).
        inv_homog = jnp.linalg.inv(homog)
        for fr2 in range(pts.shape[0]):
          res_homog[fr2] = inv_homog @ res_homog[fr2]
          _, _, tformed = compute_inliers(
              res_homog[fr2], thresh, canonical_pts, pts[fr2]
          )
          all_tformed_pts[fr] = tformed
          homog = np.eye(3)
        canonical_pts, _ = compute_canonical_points(
            all_tformed_pts,
            all_tformed_invalid,
            all_err,
            thresh,
            outer_thresh,
            outlier_point_threshold,
        )
      _, err, tformed = compute_inliers(homog, thresh, canonical_pts, pts[fr])
      all_tformed_pts[fr] = tformed
      all_err[fr] = err
      res_homog[fr] = homog
      canonical_pts, canonical_invalid = compute_canonical_points(
          all_tformed_pts,
          all_tformed_invalid,
          err,
          thresh,
          outer_thresh,
          outlier_point_threshold,
      )

  all_err = jnp.transpose(all_err)
  scaler = np.array(list(image_dimensions)+[1,])
  res_homog = res_homog @ np.diag((1.0 / scaler))

  return np.stack(res_homog, axis=0), all_err, canonical_pts


def plot_tracks_tails(
    rgb, points, occluded, homogs, point_size=12, linewidth=1.5
):
  """Plot rainbow tracks with matplotlib.

  Points nearby in the points array will be assigned similar colors.  It's a
  good idea to sort them in some meaningful way before using this, e.g. by
  height.

  Args:
    rgb: rgb pixels of shape [num_frames, height, width, 3], float or uint8.
    points: Points array, float32, of shape [num_points, num_frames, 2] in x,y
      order in raster coordinates.
    occluded: Array of occlusion values, where 1 is occluded and 0 is not, of
      shape [num_points, num_frames].
    homogs: [num_frames, 3, 3] float tensor such that inv(homogs[i]) @ homogs[j]
      is a matrix that will map background points from frame j to frame i.
    point_size: to control the scale of the points.  Passed to plt.scatter.
    linewidth: to control the line thickness.  Passed to matplotlib
      LineCollection.

  Returns:
    frames: rgb frames with rendered rainbow tracks.
  """
  disp = []
  cmap = plt.cm.hsv  # pytype: disable=module-attr

  z_list = np.arange(points.shape[0])

  colors = cmap(z_list / (np.max(z_list) + 1))
  figure_dpi = 64

  figs = []
  for i in range(rgb.shape[0]):
    print(f'Plotting frame {i}...')
    fig = plt.figure(
        figsize=(rgb.shape[2] / figure_dpi, rgb.shape[1] / figure_dpi),
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
    points = np.maximum(points, 0.0)
    points = np.minimum(points, [rgb.shape[2], rgb.shape[1]])
    plt.scatter(points[:, i, 0], points[:, i, 1], s=point_size, c=colalpha)
    reference = points[:, i]
    reference_occ = occluded[:, i : i + 1]
    for j in range(i - 1, -1, -1):
      points_homo = np.concatenate(
          [points[:, j], np.ones_like(points[:, j, 0:1])], axis=1
      )
      points_transf = np.transpose(
          np.matmul(
              np.matmul(np.linalg.inv(homogs[i]), homogs[j]),
              np.transpose(points_homo),
          )
      )
      points_transf = points_transf[:, :2] / (
          np.maximum(1e-12, np.abs(points_transf[:, 2:]))
          * np.sign(points_transf[:, 2:])
      )

      pts = np.stack([points_transf, reference], axis=1)
      oof = jnp.logical_or(
          pts < 1.0, pts > np.array([rgb.shape[2], rgb.shape[1]])
      )
      oof = np.logical_or(oof[:, 0], oof[:, 1])
      oof = np.logical_or(oof[:, 0:1], oof[:, 1:2])

      pts = np.maximum(pts, 1.0)
      pts = np.minimum(pts, np.array([rgb.shape[2], rgb.shape[1]]) - 1)
      colalpha2 = np.concatenate(
          [
              colors[:, :-1],
              (1 - occluded[:, j : j + 1]) * (1 - reference_occ) * (1 - oof),
          ],
          axis=1,
      )
      reference_occ = occluded[:, j : j + 1]

      plt.gca().add_collection(
          matplotlib.collections.LineCollection(
              pts, color=colalpha2, linewidth=linewidth
          )
      )
      reference = points_transf

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

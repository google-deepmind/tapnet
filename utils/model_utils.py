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

"""Utilities and losses for building and training TAP models."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tapnet.utils import transforms


def huber_loss(
    tracks: chex.Array, target_points: chex.Array, occluded: chex.Numeric
) -> chex.Array:
  """Huber loss for point trajectories."""
  error = tracks - target_points
  # Huber loss with a threshold of 4 pixels
  distsqr = jnp.sum(jnp.square(error), axis=-1)
  dist = jnp.sqrt(distsqr + 1e-12)  # add eps to prevent nan
  delta = 4.0
  loss_huber = jnp.where(
      dist < delta, distsqr / 2, delta * (jnp.abs(dist) - delta / 2)
  )
  loss_huber *= 1.0 - occluded

  loss_huber = jnp.mean(loss_huber, axis=[1, 2])

  return loss_huber


def prob_loss(
    tracks: chex.Array,
    expd: chex.Array,
    target_points: chex.Array,
    occluded: chex.Array,
    expected_dist_thresh: float = 8.0,
):
  """Loss for classifying if a point is within pixel threshold of its target."""
  # Points with an error larger than 8 pixels are likely to be useless; marking
  # them as occluded will actually improve Jaccard metrics and give
  # qualitatively better results.
  err = jnp.sum(jnp.square(tracks - target_points), axis=-1)
  invalid = (err > expected_dist_thresh**2).astype(expd.dtype)
  logprob = optax.sigmoid_binary_cross_entropy(expd, invalid)
  logprob *= 1.0 - occluded
  logprob = jnp.mean(logprob, axis=[1, 2])
  return logprob


def interp(x: chex.Array, y: chex.Array, mode: str = 'nearest') -> chex.Array:
  """Bilinear interpolation.

  Args:
    x: Grid of features to be interpolated, of shape [height, width]
    y: Points to be interpolated, of shape [num_points, 2], where each point is
      [y, x] in pixel coordinates, or [num_points, 3], where each point is [z,
      y, x].  Note that x and y are assumed to be raster coordinates: i.e. (0,
      0) refers to the upper-left corner of the upper-left pixel. z, however, is
      assumed to be frame coordinates, so 0 is the first frame, and 0.5 is
      halfway between the first and second frames.
    mode: mode for dealing with samples outside the range, passed to
      jax.scipy.ndimage.map_coordinates.

  Returns:
    The interpolated value, of shape [num_points].
  """
  # If the coordinate format is [z,y,x], we need to handle the z coordinate
  # differently per the docstring.
  if y.shape[-1] == 3:
    y = jnp.concatenate([y[..., 0:1], y[..., 1:] - 0.5], axis=-1)
  else:
    y = y - 0.5

  return jax.scipy.ndimage.map_coordinates(
      x,
      jnp.transpose(y),
      order=1,
      mode=mode,
  )


def soft_argmax_heatmap(
    softmax_val: chex.Array,
    threshold: chex.Numeric = 5,
) -> chex.Array:
  """Computes the soft argmax a heatmap.

  Finds the argmax grid cell, and then returns the average coordinate of
  surrounding grid cells, weighted by the softmax.

  Args:
    softmax_val: A heatmap of shape [height, width], containing all positive
      values summing to 1 across the entire grid.
    threshold: The radius of surrounding cells to consider when computing the
      average.

  Returns:
    The soft argmax, which is a single point [x,y] in grid coordinates.
  """
  x, y = jnp.meshgrid(
      jnp.arange(softmax_val.shape[1]),
      jnp.arange(softmax_val.shape[0]),
  )
  coords = jnp.stack([x + 0.5, y + 0.5], axis=-1)
  argmax_pos = jnp.argmax(jnp.reshape(softmax_val, -1))
  pos = jnp.reshape(coords, [-1, 2])[argmax_pos, jnp.newaxis, jnp.newaxis, :]
  valid = jnp.sum(
      jnp.square(coords - pos),
      axis=-1,
      keepdims=True,
  ) < jnp.square(threshold)
  weighted_sum = jnp.sum(
      coords * valid * softmax_val[:, :, jnp.newaxis],
      axis=(0, 1),
  )
  sum_of_weights = jnp.maximum(
      jnp.sum(valid * softmax_val[:, :, jnp.newaxis], axis=(0, 1)),
      1e-12,
  )
  return weighted_sum / sum_of_weights


def heatmaps_to_points(
    all_pairs_softmax: chex.Array,
    image_shape: chex.Shape,
    threshold: chex.Numeric = 5,
    query_points: Optional[chex.Array] = None,
) -> chex.Array:
  """Given a batch of heatmaps, compute a soft argmax.

  If query points are given, constrain that the query points are returned
  verbatim.

  Args:
    all_pairs_softmax: A set of heatmaps, of shape [batch, num_points, time,
      height, width].
    image_shape: The shape of the original image that the feature grid was
      extracted from.  This is needed to properly normalize coordinates.
    threshold: Threshold for the soft argmax operation.
    query_points (optional): If specified, we assume these points are given as
      ground truth and we reproduce them exactly.  This is a set of points of
      shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
      raster coordinates.

  Returns:
    predicted points, of shape [batch, num_points, time, 2], where each point is
      [x, y] in raster coordinates.  These are the result of a soft argmax ecept
      where the query point is specified, in which case the query points are
      returned verbatim.
  """
  # soft_argmax_heatmap operates over a single heatmap.  We vmap it across
  # batch, num_points, and frames.
  vmap_sah = soft_argmax_heatmap
  for _ in range(3):
    vmap_sah = jax.vmap(vmap_sah, (0, None))
  out_points = vmap_sah(all_pairs_softmax, threshold)

  feature_grid_shape = all_pairs_softmax.shape[1:]
  # Note: out_points is now [x, y]; we need to divide by [width, height].
  # image_shape[3] is width and image_shape[2] is height.
  out_points = transforms.convert_grid_coordinates(
      out_points,
      feature_grid_shape[3:1:-1],
      image_shape[3:1:-1],
  )
  assert feature_grid_shape[1] == image_shape[1]
  if query_points is not None:
    # The [..., 0:1] is because we only care about the frame index.
    query_frame = transforms.convert_grid_coordinates(
        query_points,
        image_shape[1:4],
        feature_grid_shape[1:4],
        coordinate_format='tyx',
    )[..., 0:1]
    query_frame = jnp.array(jnp.round(query_frame), jnp.int32)
    frame_indices = jnp.arange(image_shape[1], dtype=jnp.int32)[
        jnp.newaxis, jnp.newaxis, :
    ]
    is_query_point = query_frame == frame_indices

    is_query_point = is_query_point[:, :, :, jnp.newaxis]
    out_points = (
        out_points * (1 - is_query_point)
        + query_points[:, :, jnp.newaxis, 2:0:-1] * is_query_point
    )

  return out_points


def generate_default_resolutions(full_size, train_size, num_levels=None):
  """Generate a list of logarithmically-spaced resolutions.

  Generated resolutions are between train_size and full_size, inclusive, with
  num_levels different resolutions total.  Useful for generating the input to
  refinement_resolutions in PIPs.

  Args:
    full_size: 2-tuple of ints.  The full image size desired.
    train_size: 2-tuple of ints.  The smallest refinement level.  Should
      typically match the training resolution, which is (256, 256) for TAPIR.
    num_levels: number of levels.  Typically each resolution should be less than
      twice the size of prior resolutions.

  Returns:
    A list of resolutions.
  """
  if all([x == y for x, y in zip(train_size, full_size)]):
    return [train_size]

  if num_levels is None:
    size_ratio = np.array(full_size) / np.array(train_size)
    num_levels = int(np.ceil(np.max(np.log2(size_ratio))) + 1)

  if num_levels <= 1:
    return [train_size]

  h, w = full_size[0:2]
  if h % 8 != 0 or w % 8 != 0:
    print(
        'Warning: output size is not a multiple of 8. Final layer '
        + 'will round size down.'
    )
  ll_h, ll_w = train_size[0:2]

  sizes = []
  for i in range(num_levels):
    size = (
        int(round((ll_h * (h / ll_h) ** (i / (num_levels - 1))) // 8)) * 8,
        int(round((ll_w * (w / ll_w) ** (i / (num_levels - 1))) // 8)) * 8,
    )
    sizes.append(size)
  return sizes


def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (
      1 - jax.nn.sigmoid(expected_dist)
  ) > 0.5
  return visibles

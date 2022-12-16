# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TAP-Net model definition."""

import functools
from typing import Optional, Mapping, Tuple

import chex
from einshape import jax_einshape as einshape
import haiku as hk
import jax
import jax.numpy as jnp

from tapnet.models import tsm_resnet
from tapnet.utils import transforms

# (num_frames, height, width, channels)
TRAIN_SIZE = (24, 256, 256, 3)


def interp(x: chex.Array, y: chex.Array) -> chex.Array:
  """Bilinear interpolation.

  Args:
    x: Grid of features to be interpolated, of shape [height, width]
    y: Points to be interpolated, of shape [num_points, 2], where each point is
      [y, x] in pixel coordinates, or [num_points, 3], where each point is
      [z, y, x].  Note that x and y are assumed to be raster coordinates:
      i.e. (0, 0) refers to the upper-left corner of the upper-left pixel.
      z, however, is assumed to be frame coordinates, so 0 is the first frame,
      and 0.5 is halfway between the first and second frames.

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
      mode='nearest',
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
  valid = (
      jnp.sum(
          jnp.square(coords - pos),
          axis=-1,
          keepdims=True,
      ) < jnp.square(threshold))
  weighted_sum = jnp.sum(
      coords * valid * softmax_val[:, :, jnp.newaxis],
      axis=(0, 1),
  )
  sum_of_weights = (
      jnp.maximum(
          jnp.sum(valid * softmax_val[:, :, jnp.newaxis], axis=(0, 1)),
          1e-12,
      ))
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
      shape [batch, num_points, 3], where each entry is [t, y, x] normalized
      between -1 and 1.

  Returns:
    predicted points, of shape [batch, num_points, time, 2], where each point is
      [x, y] normalized between -1 and 1.  These are the result of a soft
      argmax except where the query point is specified, in which case the query
      points are returned verbatim.
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
    is_query_point = jnp.equal(
        jnp.array(jnp.round(query_frame), jnp.int32),
        jnp.arange(image_shape[1], dtype=jnp.int32)[jnp.newaxis,
                                                    jnp.newaxis, :],
    )
    out_points = out_points * (
        1.0 - is_query_point[:, :, :, jnp.newaxis]
    ) + query_points[:, :, jnp.newaxis, 2:0:-1] * is_query_point[:, :, :,
                                                                 jnp.newaxis]
  return out_points


def create_batch_norm(x: chex.Array, is_training: bool) -> chex.Array:
  """Function to allow TSM-ResNet to create batch norm layers."""
  return hk.BatchNorm(
      create_scale=True,
      create_offset=True,
      decay_rate=0.9,
      cross_replica_axis='i',
  )(x, is_training)


class TAPNet(hk.Module):
  """Joint model for performing flow-based tasks."""

  def __init__(
      self,
      feature_grid_stride: int = 8,
  ):
    """Initialize the model and provide kwargs for the various components.

    Args:
      feature_grid_stride: Stride to extract features.  For TSM-ResNet,
        supported values are 8 (default), 16, and 32.
    """

    super().__init__()

    self.feature_grid_stride = feature_grid_stride
    self.softmax_temperature = 10.0

    self.tsm_resnet = tsm_resnet.TSMResNetV2(
        normalize_fn=create_batch_norm,
        num_frames=TRAIN_SIZE[0],
        channel_shift_fraction=[0.125, 0.125, 0., 0.],
        name='tsm_resnet_video',
    )

    self.cost_volume_track_mods = {
        'hid1':
            hk.Conv3D(
                16,
                [1, 3, 3],
                name='cost_volume_regression_1',
                stride=[1, 1, 1],
            ),
        'hid2':
            hk.Conv3D(
                1,
                [1, 3, 3],
                name='cost_volume_regression_2',
                stride=[1, 1, 1],
            ),
        'hid3':
            hk.Conv3D(
                32,
                [1, 3, 3],
                name='cost_volume_occlusion_1',
                stride=[1, 2, 2],
            ),
        'hid4':
            hk.Linear(16, name='cost_volume_occlusion_2'),
        'occ_out':
            hk.Linear(1, name='occlusion_out'),
        'regression_hid':
            hk.Linear(128, name='regression_hid'),
        'regression_out':
            hk.Linear(2, name='regression_out'),
    }

  def tracks_from_cost_volume(
      self,
      interp_feature_heads: chex.Array,
      feature_grid_heads: chex.Array,
      query_points: Optional[chex.Array],
      im_shp: Optional[chex.Shape] = None,
  ) -> Tuple[chex.Array, chex.Array]:
    """Converts features into tracks by computing a cost volume.

    The computed cost volume will have shape
      [batch, num_queries, time, height, width, num_heads], which can be very
      memory intensive.

    Args:
      interp_feature_heads: A tensor of features for each query point, of shape
        [batch, num_queries, channels, heads].
      feature_grid_heads: A tensor of features for the video, of shape [batch,
        time, height, width, channels, heads].
      query_points: When computing tracks, we assume these points are given as
        ground truth and we reproduce them exactly.  This is a set of points of
        shape [batch, num_points, 3], where each entry is [t, y, x] normalized
        between -1 and 1.
      im_shp: The shape of the original image, i.e., [batch, num_frames, time,
        height, width, 3].

    Returns:
      A 2-tuple of the inferred points (of shape
        [batch, num_points, num_frames, 2] where each point is [x, y]) and
        inferred occlusion (of shape [batch, num_points, num_frames], where
        each is a logit where higher means occluded)
    """

    mods = self.cost_volume_track_mods
    # Note: time is first axis to prevent the TPU from padding
    cost_volume = jnp.einsum(
        'bncd,bthwcd->tbnhwd',
        interp_feature_heads,
        feature_grid_heads,
    )
    shape = cost_volume.shape
    cost_volume = einshape('tbnhwd->t(bn)hwd', cost_volume)

    occlusion = mods['hid1'](cost_volume)
    occlusion = jax.nn.relu(occlusion)

    pos = mods['hid2'](occlusion)
    pos = jax.nn.softmax(pos * self.softmax_temperature, axis=(-2, -3))
    pos = einshape('t(bn)hw1->bnthw', pos, n=shape[2])
    points = heatmaps_to_points(pos, im_shp, query_points=query_points)

    occlusion = mods['hid3'](occlusion)
    occlusion = jnp.mean(occlusion, axis=(-2, -3))
    occlusion = mods['hid4'](occlusion)
    occlusion = jax.nn.relu(occlusion)
    occlusion = mods['occ_out'](occlusion)
    occlusion = jnp.transpose(occlusion, (1, 0, 2))
    assert occlusion.shape[1] == shape[0]
    occlusion = jnp.reshape(occlusion, (shape[1], shape[2], shape[0]))
    return points, occlusion

  def __call__(
      self,
      video: chex.Array,
      is_training: bool,
      query_points: chex.Array,
      compute_regression: bool = True,
      query_chunk_size: Optional[int] = None,
      get_query_feats: bool = False,
      feature_grid: Optional[chex.Array] = None,
  ) -> Mapping[str, chex.Array]:
    """Runs a forward pass of the model.

    Args:
      video: A 4-D or 5-D tensor representing a batch of sequences of images. In
        the 4-D case, we assume the entire batch has been concatenated along the
        batch dimension, one sequence after the other.  This can speed up
        inference on the TPU and save memory.
      is_training: Whether we are training.
      query_points: The query points for which we compute tracks.
      compute_regression: if True, compute tracks using cost volumes; otherwise
        simply compute features (required for the baseline)
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      get_query_feats: If True, also return the features for each query obtained
        using bilinear interpolation from the feature grid
      feature_grid: If specified, use this as the feature grid rather than
        computing it from the pixels.

    Returns:
      A dict of outputs, including:
        feature_grid: a TSM-ResNet feature grid of shape
          [batch, num_frames, height//stride, width//stride, channels]
        query_feats (optional): A feature for each query point, of size
          [batch, num_queries, channels]
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          scaled to the range [-1, 1]
    """
    num_frames = None
    if feature_grid is None:
      latent = self.tsm_resnet(
          video,
          is_training=is_training,
          output_stride=self.feature_grid_stride,
          out_num_frames=num_frames,
          final_endpoint='tsm_resnet_unit_2',
      )

      feature_grid = latent / jnp.sqrt(
          jnp.maximum(
              jnp.sum(jnp.square(latent), axis=-1, keepdims=True),
              1e-12,
          ))

    shape = video.shape
    if num_frames is not None and len(shape) < 5:
      shape = (shape[0] // num_frames, num_frames) + shape[1:]

    # shape is [batch_size, time, height, width, channels]; conversion needs
    # [time, width, height]
    position_in_grid = transforms.convert_grid_coordinates(
        query_points,
        shape[1:4],
        feature_grid.shape[1:4],
        coordinate_format='tyx',
    )
    interp_features = jax.vmap(
        jax.vmap(
            interp,
            in_axes=(3, None),
            out_axes=1,
        ))(feature_grid, position_in_grid)
    num_heads = 1
    feature_grid_heads = einshape('bthw(cd)->bthwcd', feature_grid, d=num_heads)
    interp_features_heads = einshape(
        'bn(cd)->bncd',
        interp_features,
        d=num_heads,
    )
    out = {'feature_grid': feature_grid}
    if get_query_feats:
      out['query_feats'] = interp_features

    if compute_regression:
      assert query_chunk_size is not None
      all_occ = []
      all_pts = []
      infer = functools.partial(self.tracks_from_cost_volume, im_shp=shape)

      for i in range(0, query_points.shape[1], query_chunk_size):
        points, occlusion = infer(
            interp_features_heads[:, i:i + query_chunk_size],
            feature_grid_heads,
            query_points[:, i:i + query_chunk_size],
        )
        all_occ.append(occlusion)
        all_pts.append(points)
      occlusion = jnp.concatenate(all_occ, axis=1)
      points = jnp.concatenate(all_pts, axis=1)

      out['occlusion'] = occlusion
      out['tracks'] = points

    return out

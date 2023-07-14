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

"""TAP-Net model definition."""

import functools
from typing import Optional, Mapping, Tuple

import chex
from einshape import jax_einshape as einshape
import haiku as hk
import jax
import jax.numpy as jnp

from tapnet.models import tsm_resnet
from tapnet.utils import model_utils
from tapnet.utils import transforms


def create_batch_norm(
    x: chex.Array, is_training: bool, cross_replica_axis: Optional[str]
) -> chex.Array:
  """Function to allow TSM-ResNet to create batch norm layers."""
  return hk.BatchNorm(
      create_scale=True,
      create_offset=True,
      decay_rate=0.9,
      cross_replica_axis=cross_replica_axis,
  )(x, is_training)


class TAPNet(hk.Module):
  """Joint model for performing flow-based tasks."""

  def __init__(
      self,
      feature_grid_stride: int = 8,
      num_heads: int = 1,
      cross_replica_axis: Optional[str] = 'i',
      num_frames: int = 24,
  ):
    """Initialize the model and provide kwargs for the various components.

    Args:
      feature_grid_stride: Stride to extract features.  For TSM-ResNet,
        supported values are 8 (default), 16, and 32.
      num_heads: Number of heads in the cost volume.
      cross_replica_axis: Which cross replica axis to use for the batch norm.
      num_frames: Number of frames passed into TSM-ResNet.
    """

    super().__init__()

    self.feature_grid_stride = feature_grid_stride
    self.num_heads = num_heads
    self.softmax_temperature = 10.0

    self.tsm_resnet = tsm_resnet.TSMResNetV2(
        normalize_fn=functools.partial(
            create_batch_norm, cross_replica_axis=cross_replica_axis
        ),
        num_frames=num_frames,
        channel_shift_fraction=[0.125, 0.125, 0.0, 0.0],
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
        shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
        raster coordinates.
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
    points = model_utils.heatmaps_to_points(
        pos, im_shp, query_points=query_points
    )

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
          in raster coordinates
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
            model_utils.interp,
            in_axes=(3, None),
            out_axes=1,
        )
    )(feature_grid, position_in_grid)
    feature_grid_heads = einshape(
        'bthw(cd)->bthwcd', feature_grid, d=self.num_heads
    )
    interp_features_heads = einshape(
        'bn(cd)->bncd',
        interp_features,
        d=self.num_heads,
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

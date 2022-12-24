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

"""Perceiver task module."""

import functools
from os import path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jaxline import utils
import matplotlib
import matplotlib.pyplot as plt
import mediapy as media
from ml_collections import config_dict
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from tapnet import evaluation_datasets
from tapnet import tapnet_model
from tapnet import task
from tapnet.data import viz_utils
from tapnet.utils import transforms

matplotlib.use('Agg')


def sigmoid_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    reduction: Optional[str] = None,
) -> chex.Array:
  """Computes sigmoid cross entropy given logits and multiple class labels."""
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  loss = -labels * log_p - (1.0 - labels) * log_not_p
  result = jnp.asarray(loss)
  if reduction:
    if reduction == 'mean':
      result = jnp.mean(result)
    else:
      raise ValueError(f'Wrong reduction name {reduction}')
  return result


def huber_loss(
    tracks: chex.Array,
    target_points: chex.Array,
    occluded: chex.Numeric,
) -> chex.Array:
  """Huber loss for point trajectories."""
  error = tracks - target_points
  # Huber loss with a threshold of 4 pixels
  distsqr = jnp.sum(jnp.square(error), axis=-1)
  dist = jnp.sqrt(distsqr + 1e-12)  # add eps to prevent nan
  delta = 4.0
  loss_huber = jnp.where(
      dist < delta,
      distsqr / 2,
      delta * (jnp.abs(dist) - delta / 2),
  )
  loss_huber *= 1.0 - occluded

  loss_huber = jnp.mean(loss_huber, axis=[1, 2])

  return loss_huber


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
        [colors[:, :-1], 1 - occluded[:, i : i + 1]],
        axis=1,
    )
    plt.scatter(points[:, i, 0], points[:, i, 1], s=3, c=colalpha)
    occ2 = occluded[:, i : i + 1]
    if gt_occluded is not None:
      occ2 *= 1 - gt_occluded[:, i : i + 1]
    colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

    plt.scatter(
        points[:, i, 0],
        points[:, i, 1],
        s=20,
        facecolors='none',
        edgecolors=colalpha,
    )
    if gt_points is not None:
      colalpha = np.concatenate(
          [colors[:, :-1], 1 - gt_occluded[:, i : i + 1]], axis=1
      )
      plt.scatter(
          gt_points[:, i, 0],
          gt_points[:, i, 1],
          s=15,
          c=colalpha,
          marker='D',
      )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype='uint8',
    ).reshape(int(height), int(width), 3)
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

    video_frames = plot_tracks_v3(
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
        fps=10,
        codec='h264',
        bps=400000,
    ) as video_writer:
      for j in range(video_frames.shape[0]):
        fr = video_frames[j]
        video_writer.add_image(fr.astype(np.uint8))


class SupervisedPointPrediction(task.Task):
  """A task for predicting point tracks and training on ground-truth.

  This task has a simple forward pass which predicts points, which is
  also used for running evaluators on datasets where ground-truth is
  known.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict,
      input_key: str = 'kubric',
      contrastive_loss_weight: float = 0.05,
      prediction_algo: str = 'cost_volume_regressor',
  ):
    """Constructs a task for supervised learning on Kubric.

    Args:
      config: a ConfigDict for configuring this experiment, notably including
        the paths for checkpoints and datasets.
      input_key: The forward pass takes an input dict.  Inference or learning
        will be performed on input[input_key]['video']
      contrastive_loss_weight: Weight for the additional contrastive loss that's
        applied alongside the trajectory prediction loss.
      prediction_algo: specifies the network architecture to use to make
        predictions.  Can be 'cost_volume_regressor' for the algorithm presented
        in the TAPNet paper, or 'cost_volume_cycle_consistency' for the VFS-Like
        algorithm presented in the earlier Kubric paper.
    """

    super().__init__()

    self.input_key = input_key
    self.prediction_algo = prediction_algo
    self.contrastive_loss_weight = contrastive_loss_weight
    self.config = config
    self.softmax_temperature = 10.0

  def forward_fn(
      self,
      inputs: chex.ArrayTree,
      is_training: bool,
      shared_modules: Optional[Mapping[str, task.SharedModule]] = None,
      input_key: Optional[str] = None,
      query_chunk_size: int = 16,
      get_query_feats: bool = True,
  ) -> Mapping[str, chex.Array]:
    """Forward pass for predicting point tracks.

    Args:
      inputs: Input dict.  Inference will be performed on will be performed on
        inputs[input_key]['video'] (with fallback to the input_key specified in
        the constructor).  Input videos should be a standard video tensor
        ([batch, num_frames, height, width, 3]) normalize to [-1,1].
        inputs[input_key]['query_points'] specifies the query point locations,
        of shape [batch, num_queries, 3], where each query is [t,y,x]
        coordinates in frame/raster coordinates.
      is_training: Is the model in training mode.
      shared_modules: Haiku modules, injected by experiment.py.
        shared_modules['tapnet_model'] should be a TAPNetModel.
      input_key: Run on inputs[input_key]['video']. If None, use the input_key
        from the constructor.
      query_chunk_size: Compute predictions on this many queries simultaneously.
        This saves memory as the cost volumes can be very large.
      get_query_feats: If True, also return features for each query.

    Returns:
      Result dict produced by calling the tapnet model. See tapnet_model.py.
    """
    if input_key is None:
      input_key = self.input_key
    frames = inputs[input_key]['video']

    if self.prediction_algo in [
        'cost_volume_regressor',
        'cost_volume_cycle_consistency',
    ]:
      return shared_modules['tapnet_model'](
          frames,
          is_training=is_training,
          query_points=inputs[input_key]['query_points'],
          query_chunk_size=query_chunk_size,
          get_query_feats=get_query_feats,
      )
    else:
      raise ValueError('Unsupported prediction_algo:' + self.prediction_algo)

  def _loss_fn(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      wrapped_forward_fn: task.WrappedForwardFn,
      is_training: bool = True,
      input_key: Optional[str] = None,
  ):
    """Loss function, used for training, depending on the algorithm.

    This includes the Huber and softmax cross entropy losses for cost volume
    regression, plus the contrastive loss for cost volume regression and
    the baseline cost volume cycle consistency.

    Args:
      params: hk.Params with the model parameters
      state: hk.State with the model state
      inputs: Input dict.  Inference will be performed on will be performed on
        inputs[input_key]['video'] (with fallback to the input_key specified in
        the constructor).  Input videos should be a standard video tensor
        ([batch, num_frames, height, width, 3]) normalize to [-1,1].
        inputs[input_key]['query_points'] specifies the query point locations,
        of shape [batch, num_queries, 3], where each query is [t,y,x]
        coordinates in frame/raster coordinates.
        inputs[input_key]['target_points'] is the ground-truth locations on each
        frame, of shape [batch, num_queries, num_frames, 2], where each point is
        [x,y] raster coordinates (in the range [0,width]/[0,height]).
        inputs[input_key]['occluded'] is the ground-truth occlusion flag, a
        boolean of shape [batch, num_queries, num_frames], where True indicates
        occluded.
      rng: jax.random.PRNGKey for random number generation.
      wrapped_forward_fn: A wrapper around self.forward which can inject Haiku
        parameters.
      is_training: Is the model in training mode.
      input_key: Run on inputs[input_key]['video']. If None, use the input_key
        from the constructor.

    Returns:
      A 2-tuple consisting of the summed loss, and a 2-tuple containing scalar
        outputs and the updated state.  The loss scalars are broken down into
        the position loss, occlusion loss, and contrastive loss.
    """
    if input_key is None:
      input_key = self.input_key

    query_chunk_size = 16

    output, state = functools.partial(
        wrapped_forward_fn,
        input_key=input_key,
        query_chunk_size=query_chunk_size,
    )(params, state, rng, inputs, is_training=is_training)
    loss_scalars = {}
    loss = 0
    if self.prediction_algo in ['cost_volume_regressor']:
      loss_huber = huber_loss(
          output['tracks'],
          inputs[input_key]['target_points'],
          inputs[input_key]['occluded'],
      )
      # For the original experiments, the loss was defined on coordinates in the
      # range [-1, 1], so convert them into that scale.
      loss_huber = loss_huber * 4.0 / np.prod(tapnet_model.TRAIN_SIZE[1:3])
      loss_huber = jnp.mean(loss_huber)

      loss_occ = sigmoid_cross_entropy(
          output['occlusion'],
          jnp.array(inputs[input_key]['occluded'], jnp.float32),
          reduction='mean',
      )

      loss = loss_huber * 100.0 + loss_occ

      loss_scalars = dict(position_loss=loss_huber, occlusion_loss=loss_occ)
    if self.prediction_algo in [
        'cost_volume_cycle_consistency',
    ]:
      feature_grid = output['feature_grid']

      query_feats = output['query_feats']

      loss_contrast = []

      # This computes the contrastive loss from the paper.  We break the set of
      # queries up into chunks in order to save memory.
      for qchunk in range(0, query_feats.shape[1], query_chunk_size):
        qchunk_lo = qchunk
        qchunk_hi = qchunk + query_chunk_size
        im_shp = inputs[input_key]['video'].shape
        all_pairs_dots = jnp.einsum(
            'bnc,bthwc->bnthw',
            query_feats[:, qchunk_lo:qchunk_hi],
            feature_grid,
        )
        all_pairs_softmax = jax.nn.log_softmax(
            all_pairs_dots * self.softmax_temperature,
            axis=(2, 3, 4),
        )
        im_shp = inputs[input_key]['video'].shape
        point_track = inputs[input_key]['target_points'][
            :, qchunk_lo:qchunk_hi, ..., :
        ]
        position_in_grid2 = transforms.convert_grid_coordinates(
            point_track,
            im_shp[3:1:-1],
            feature_grid.shape[3:1:-1],
        )

        # result is shape [batch, num_queries, time]
        # Interp handles a single 2D slice.  We need to vmap it across all
        # batch, queries, and time to extract the softmax value associated with
        # the entire trajectory.
        #
        # Note: grid positions are in [x, y] format, but interp needs
        # coordinates in [y, x] format.
        interp_softmax = jax.vmap(jax.vmap(jax.vmap(tapnet_model.interp)))(
            all_pairs_softmax,
            position_in_grid2[..., ::-1],
        )

        loss_contrast.append(
            jnp.mean(
                interp_softmax
                * (1.0 - inputs[input_key]['occluded'][:, qchunk_lo:qchunk_hi]),
                axis=-1,
            )
        )

      loss_contrast = -jnp.mean(jnp.concatenate(loss_contrast, 1))
      loss += loss_contrast * self.contrastive_loss_weight
      loss_scalars['loss_contrast'] = loss_contrast

    loss_scalars['loss'] = loss
    scaled_loss = loss / jax.device_count()

    return scaled_loss, (loss_scalars, state)

  def get_gradients(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      global_step: chex.Array,
      wrapped_forward_fn: task.WrappedForwardFn,
      is_training: bool = True,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, Mapping[str, chex.Array]]:
    """Gets the gradients for the loss function.  See _loss_fn."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (loss_scalars, state) = grad_loss_fn(
        params,
        state,
        inputs,
        rng,
        wrapped_forward_fn,
        is_training=is_training,
    )
    grads = jax.lax.psum(scaled_grads, axis_name='i')
    scalars = {}
    scalars.update(loss_scalars)
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return grads, state, scalars

  def evaluate(
      self,
      global_step: chex.Array,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      rng: chex.PRNGKey,
      wrapped_forward_fn: task.WrappedForwardFn,
      mode: str,
  ) -> Mapping[str, chex.Array]:
    """Run an evaluation epoch.  See base class."""
    global_step = np.array(utils.get_first(global_step))
    if mode == 'eval_inference':
      scalars = jax.device_get(
          self._eval_inference(
              global_step,
              utils.get_first(state),
              utils.get_first(params),
              utils.get_first(rng),
              wrapped_forward_fn,
          )
      )
    else:
      scalars = jax.device_get(
          self._eval_epoch(
              global_step,
              utils.get_first(state),
              utils.get_first(params),
              utils.get_first(rng),
              wrapped_forward_fn,
              mode,
          )
      )

      logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def _infer_batch(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      wrapped_forward_fn: task.WrappedForwardFn,
      input_key: Optional[str] = None,
      query_chunk_size: int = 16,
  ) -> Tuple[chex.Array, chex.Array, Mapping[str, chex.Array]]:
    """Runs inference on a single batch and compute metrics.

    For cost_volume_regressor we return the outputs directly inferred from the
    model.  For cost_volume_cycle_consistency, we compute the tracks by
    computing the soft argmax operation (which tapnet_model.py doesn't compute)
    and then use cycle-consistency to infer occlusion.

    Args:
      params: hk.Params with the model parameters
      state: hk.State with the model state
      inputs: Input dict.  Inference will be performed on will be performed on
        inputs[input_key]['video'] (with fallback to the input_key specified in
        the constructor).  Input videos should be a standard video tensor
        ([batch, num_frames, height, width, 3]) normalize to [-1,1].
        inputs[input_key]['query_points'] specifies the query point locations,
        of shape [batch, num_queries, 3], where each query is [t,y,x]
        coordinates in frame/raster coordinates.
        inputs[input_key]['target_points'] is the ground-truth locations on each
        frame, of shape [batch, num_queries, num_frames, 2], where each point is
        [x,y] raster coordinates (in the range [0,width]/[0,height]).
        inputs[input_key]['occluded'] is the ground-truth occlusion flag, a
        boolean of shape [batch, num_queries, num_frames], where True indicates
        occluded.
      rng: jax.random.PRNGKey for random number generation.
      wrapped_forward_fn: A wrapper around self.forward_fn which can inject
        Haiku parameters.  It expects the same inputs as self.forward, plus
        Haiku parameters, state, and a jax.random.PRNGKey.  It's the result of
        applying hk.transform to self.forward_fn.
      input_key: Run on inputs[input_key]['video']. If None, use the input_key
        from the constructor.
      query_chunk_size: Run computation on this many queries at a time to save
        memory.

    Returns:
      A 3-tuple consisting of the occlusion logits, of shape
        [batch, num_queries, num_frames], the predicted position, of shape
        [batch, num_queries, num_frames, 2], and a dict of loss scalars.
    """
    # Features for each query point are required when using cycle consistency.
    get_query_feats = self.prediction_algo in ['cost_volume_cycle_consistency']
    output, _ = functools.partial(
        wrapped_forward_fn,
        input_key=input_key,
    )(
        params,
        state,
        rng,
        inputs,
        is_training=False,
        query_chunk_size=query_chunk_size,
        get_query_feats=get_query_feats,
    )
    loss_scalars = {}
    if self.prediction_algo in ['cost_volume_regressor']:
      # Outputs are already in the correct format for cost_volume_regressor.
      tracks = output['tracks']
      loss_occ = sigmoid_cross_entropy(
          output['occlusion'],
          jnp.array(inputs[input_key]['occluded'], jnp.float32),
          reduction=None,
      )
      loss_occ = jnp.mean(loss_occ, axis=(1, 2))
      occlusion = output['occlusion']
      loss_scalars['loss_occ'] = loss_occ
    else:
      # compute forward-backward cycle consistency to infer occlusions.
      feature_grid = output['feature_grid']
      query_feats = output['query_feats']

      all_tracks = []
      all_occlusion = []

      # We again chunk the queries to save memory; these einsums are big.
      for qchunk in range(0, query_feats.shape[1], query_chunk_size):
        im_shp = inputs[input_key]['video'].shape
        # Compute pairwise dot products between queries and all other features
        all_pairs_dots = jnp.einsum(
            'bnc,bthwc->bnthw',
            query_feats[:, qchunk : qchunk + query_chunk_size],
            feature_grid,
        )
        # Compute the soft argmax for each frame
        query_point_chunk = inputs[input_key]['query_points'][
            :, qchunk : qchunk + query_chunk_size
        ]
        tracks = tapnet_model.heatmaps_to_points(
            jax.nn.softmax(
                all_pairs_dots * self.softmax_temperature,
                axis=(-1, -2),
            ),
            im_shp,
            query_points=query_point_chunk,
        )

        # Extract the argmax feature from each frame for each query using
        # bilinear interpolation.
        frame_id = jnp.broadcast_to(
            jnp.arange(tracks.shape[-2])[..., jnp.newaxis],
            tracks[..., :1].shape,
        )
        position_in_grid = jnp.concatenate(
            [frame_id, tracks[..., ::-1]],
            axis=-1,
        )
        position_in_grid = transforms.convert_grid_coordinates(
            position_in_grid,
            im_shp[1:4],
            feature_grid.shape[1:4],
            coordinate_format='tyx',
        )
        # vmap over channels, duplicating the coordinates
        vmap_interp = jax.vmap(
            tapnet_model.interp, in_axes=(3, None), out_axes=1
        )
        # vmap over frames, using the same queries for each frame
        vmap_interp = jax.vmap(vmap_interp, in_axes=(None, 0), out_axes=0)
        # vmap over the batch
        vmap_interp = jax.vmap(vmap_interp)
        # interp_features is [batch_size,num_queries,num_frames,channels]
        interp_features = vmap_interp(feature_grid, position_in_grid)

        # For each query point, extract the features for the frame which
        # contains the query.
        # query_frame is [batch_size, num_queries]
        position_in_grid = jnp.concatenate(
            [frame_id, tracks[..., ::-1]],
            axis=-1,
        )
        query_frame = transforms.convert_grid_coordinates(
            inputs[input_key]['query_points'][
                :, qchunk : qchunk + query_chunk_size, ...
            ],
            im_shp[1:4],
            feature_grid.shape[1:4],
            coordinate_format='tyx',
        )[..., 0]
        query_frame = jnp.array(jnp.round(query_frame), jnp.int32)
        # target_features is [batch_size, chunk, height, width, num_channels]
        target_features = jnp.take_along_axis(
            feature_grid,
            query_frame[:, :, np.newaxis, np.newaxis, np.newaxis],
            axis=1,
        )

        # For each output point along the track, compare the features with all
        # features in the frame that the query came from
        all_pairs_dots = jnp.einsum(
            'bntc,bnhwc->bnthw',
            interp_features,
            target_features,
        )

        # Again, take the soft argmax to see if we come back to the place we
        # started from.
        # inverse_tracks is [batch_size, chunk, num_frames, 2]
        inverse_tracks = tapnet_model.heatmaps_to_points(
            jax.nn.softmax(
                all_pairs_dots * self.softmax_temperature,
                axis=(-2, -1),
            ),
            im_shp,
        )
        dist = jnp.sum(
            jnp.square(
                inverse_tracks
                - inputs[input_key]['query_points'][
                    :, qchunk : qchunk + query_chunk_size, jnp.newaxis, 2:0:-1
                ]
            ),
            axis=-1,
        )
        occlusion = dist > jnp.square(48.0)
        # We need to return logits, but the cycle consistency rule is binary.
        # So we just convert the binary values into large real values.
        occlusion = occlusion * 20.0 - 10.0

        all_occlusion.append(occlusion)
        all_tracks.append(tracks)

      tracks = jnp.concatenate(all_tracks, axis=1)
      occlusion = jnp.concatenate(all_occlusion, axis=1)

    return occlusion, tracks, loss_scalars

  def _eval_batch(
      self,
      params: chex.Array,
      state: chex.Array,
      inputs: chex.Array,
      rng: chex.PRNGKey,
      wrapped_forward_fn: task.WrappedForwardFn,
      mode: str = '',
      input_key: Optional[str] = None,
  ) -> Tuple[Mapping[str, chex.Array], Mapping[str, chex.Array]]:
    """Evaluates the model on a single batch and compute metrics.

    Args:
      params: hk.Params with the model parameters
      state: hk.State with the model state
      inputs: Input dict.  Inference will be performed on will be performed on
        inputs[input_key]['video'] (with fallback to the input_key specified in
        the constructor).  Input videos should be a standard video tensor
        ([batch, num_frames, height, width, 3]) normalize to [-1,1].
        inputs[input_key]['query_points'] specifies the query point locations,
        of shape [batch, num_queries, 3], where each query is [t,y,x]
        coordinates in frame/raster coordinates.
        inputs[input_key]['target_points'] is the ground-truth locations on each
        frame, of shape [batch, num_queries, num_frames, 2], where each point is
        [x,y] raster coordinates (in the range [0,width]/[0,height]).
        inputs[input_key]['occluded'] is the ground-truth occlusion flag, a
        boolean of shape [batch, num_queries, num_frames], where True indicates
        occluded.
      rng: jax.random.PRNGKey for random number generation.
      wrapped_forward_fn: A wrapper around self.forward_fn which can inject
        Haiku parameters.  It expects the same inputs as self.forward, plus
        Haiku parameters, state, and a jax.random.PRNGKey.  It's the result of
        applying hk.transform to self.forward_fn.
      mode: Which evaluation we're running.  For most it will compute standard
        occlusion accuracy, points within thresholds, and jaccard metrics. For
        eval_jhmdb, however, we will compute standard PCK.
      input_key: Run on inputs[input_key]['video']. If None, use the input_key
        from the constructor.

    Returns:
      A 2-tuple consisting of a dict of loss scalars and a dict of outputs.
        The latter consists of the occlusion logits, of shape
        [batch, num_queries, num_frames] and the predicted position, of shape
        [batch, num_queries, num_frames, 2].
    """
    occlusion_logits, tracks, loss_scalars = self._infer_batch(
        params,
        state,
        inputs,
        rng,
        wrapped_forward_fn,
        input_key,
        query_chunk_size=16,
    )
    loss_scalars = {**loss_scalars}  # Mutable copy.

    gt_occluded = inputs[input_key]['occluded']
    gt_target_points = inputs[input_key]['target_points']
    query_points = inputs[input_key]['query_points']

    loss_huber = huber_loss(
        tracks,
        gt_target_points,
        gt_occluded,
    )
    loss_huber = loss_huber * 4.0 / np.prod(tapnet_model.TRAIN_SIZE[1:3])
    loss_scalars['position_loss'] = loss_huber

    pred_occ = occlusion_logits > 0
    query_mode = 'first' if 'q_first' in mode else 'strided'

    metrics = evaluation_datasets.compute_tapvid_metrics(
        query_points=query_points,
        gt_occluded=gt_occluded,
        gt_tracks=gt_target_points,
        pred_occluded=pred_occ,
        pred_tracks=tracks,
        query_mode=query_mode,
    )
    loss_scalars.update(metrics)

    return loss_scalars, {'tracks': tracks, 'occlusion': occlusion_logits}

  def _build_eval_input(
      self,
      mode: str,
  ) -> Iterable[evaluation_datasets.DatasetElement]:
    """Build evalutation data reader generator.

    Args:
      mode: evaluation mode.  Can be one of 'eval_davis_points',
        'eval_robotics_points', 'eval_kinetics_points',
        'eval_davis_points_q_first', 'eval_robotics_points_q_first',
        'eval_kinetics_points_q_first', 'eval_jhmdb', 'eval_kubric',

    Yields:
      A dict with one key (for the dataset), containing a dict with the keys:
        video: Video tensor of shape [1, num_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] in pixel/raster coordinates
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] raster coordinates (in the range
          [0,width]/[0,height])
        trackgroup (optional): Index of the original track that each query
          point was sampled from.  This is useful for visualization.
        pad_extra_frames (optional): the number of pad frames that were added
          to reach num_frames.
    """
    query_mode = 'first' if 'q_first' in mode else 'strided'
    if 'eval_kubric_train' in mode:
      yield from evaluation_datasets.create_kubric_eval_train_dataset(mode)
    elif 'eval_kubric' in mode:
      yield from evaluation_datasets.create_kubric_eval_dataset(mode)
    elif 'eval_davis_points' in mode:
      yield from evaluation_datasets.create_davis_dataset(
          self.config.davis_points_path, query_mode=query_mode
      )
    elif 'eval_jhmdb' in mode:
      yield from evaluation_datasets.create_jhmdb_dataset(
          self.config.jhmdb_path
      )
    elif 'eval_robotics_points' in mode:
      yield from evaluation_datasets.create_rgb_stacking_dataset(
          self.config.robotics_points_path, query_mode=query_mode
      )
    elif 'eval_kinetics_points' in mode:
      yield from evaluation_datasets.create_kinetics_dataset(
          self.config.kinetics_points_path, query_mode=query_mode
      )
    else:
      raise ValueError(f'Unrecognized eval mode {mode}')

  def compute_pck(
      self,
      dist_all: Sequence[np.ndarray],
      dist_thresh: float,
  ) -> Sequence[float]:
    pck_all = np.zeros((len(dist_all),))
    for pidx in range(len(dist_all)):
      idxs = np.argwhere(dist_all[pidx] <= dist_thresh)
      pck = 100.0 * len(idxs) / max(1e-12, len(dist_all[pidx]))
      pck_all[pidx] = pck

    return pck_all

  def pck_evaluate(
      self,
      results: Sequence[Mapping[str, np.ndarray]],
  ) -> Mapping[str, np.ndarray]:
    num_keypoints = 15
    dist_all = [np.zeros((0, 0)) for _ in range(num_keypoints)]
    for vid_idx in range(len(results)):
      sample = results[vid_idx]

      # [2, 15, clip_len]
      pred_poses = np.transpose(sample['pred_pose'][0], (2, 0, 1))

      gt_poses = sample['gt_pose_orig'][0]
      width = sample['im_size'][1]
      height = sample['im_size'][0]

      # input is shape [15, clip_len, 2]
      invalid_x = np.logical_or(
          gt_poses[:, 0:1, 0] < 0,
          gt_poses[:, 0:1, 0] >= width,
      )
      invalid_y = np.logical_or(
          gt_poses[:, 0:1, 1] < 0,
          gt_poses[:, 0:1, 1] >= height,
      )
      invalid = np.logical_or(invalid_x, invalid_y)
      joint_visible = np.logical_not(np.tile(invalid, [1, gt_poses.shape[1]]))

      gt_poses = np.transpose(gt_poses, (2, 0, 1))

      clip_len = pred_poses.shape[-1]

      assert (
          pred_poses.shape == gt_poses.shape
      ), f'{pred_poses.shape} vs {gt_poses.shape}'

      # [15, clip_len]
      valid_max_gt_poses = gt_poses.copy()
      valid_max_gt_poses[:, ~joint_visible] = -1
      valid_min_gt_poses = gt_poses.copy()
      valid_min_gt_poses[:, ~joint_visible] = 1e6
      boxes = np.stack(
          (
              valid_max_gt_poses[0].max(axis=0)
              - valid_min_gt_poses[0].min(axis=0),
              valid_max_gt_poses[1].max(axis=0)
              - valid_min_gt_poses[1].min(axis=0),
          ),
          axis=0,
      )
      # [clip_len]
      boxes = 0.6 * np.linalg.norm(boxes, axis=0)
      for img_idx in range(clip_len):
        for t in range(num_keypoints):
          if not joint_visible[t, img_idx]:
            continue
          predx = pred_poses[0, t, img_idx]
          predy = pred_poses[1, t, img_idx]
          gtx = gt_poses[0, t, img_idx]
          gty = gt_poses[1, t, img_idx]
          dist = np.linalg.norm(np.subtract([predx, predy], [gtx, gty]))
          dist = dist / boxes[img_idx]

          dist_all[t] = np.append(dist_all[t], [[dist]])
    pck_ranges = (0.1, 0.2, 0.3, 0.4, 0.5)
    pck_all = []
    for pck_range in pck_ranges:
      pck_all.append(self.compute_pck(dist_all, pck_range))
    eval_results = {}
    for alpha, pck in zip(pck_ranges, pck_all):
      eval_results[f'PCK@{alpha}'] = np.mean(pck)

    return eval_results

  def _eval_jhmdb(
      self,
      pred_pose: chex.Array,
      gt_pose: chex.Array,
      gt_pose_orig: chex.Array,
      im_size: chex.Array,
      fname: str,
      is_first: bool = False,
  ) -> Mapping[str, np.ndarray]:
    if is_first:
      self.all_results = []
    self.all_results.append({
        'pred_pose': np.array(pred_pose),
        'gt_pose': np.array(gt_pose),
        'gt_pose_orig': np.array(gt_pose_orig),
        'im_size': np.array(im_size),
    })
    return self.pck_evaluate(self.all_results)

  def _eval_epoch(
      self,
      global_step: chex.Array,
      state: chex.ArrayTree,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      wrapped_forward_fn: task.WrappedForwardFn,
      mode: str,
  ) -> Mapping[str, chex.Array]:
    """Evaluates an epoch."""
    num_samples = 0.0
    summed_scalars = None
    batch_id = 0

    outdir = path.join(
        self.config.checkpoint_dir,
        mode,
        str(global_step),
    )

    logging.info('Saving videos to %s', outdir)

    try:
      tf.io.gfile.makedirs(outdir)
    except FileExistsError:
      print(f'Path {outdir} exists. Skip creating a new dir.')

    if 'eval_jhmdb' in mode:
      input_key = 'jhmdb'
    elif 'eval_davis_points' in mode:
      input_key = 'davis'
    elif 'eval_robotics_points' in mode:
      input_key = 'robotics'
    elif 'eval_kinetics_points' in mode:
      input_key = 'kinetics'
    else:
      input_key = 'kubric'
    eval_batch_fn = functools.partial(
        self._eval_batch,
        wrapped_forward_fn=wrapped_forward_fn,
        mode=mode,
        input_key=input_key,
    )
    for inputs in self._build_eval_input(mode):
      batch_size = inputs[input_key]['video'].shape[0]  # pytype: disable=attribute-error  # 'video' entry is array-valued
      num_samples += batch_size
      scalars, viz = eval_batch_fn(params, state, inputs, rng)
      write_viz = batch_id < 10
      if 'eval_davis_points' in mode or 'eval_robotics_points' in mode:
        # Only write videos sometimes for the small datasets; otherwise
        # there will be a crazy number of videos dumped.
        write_viz = write_viz and (global_step % 10 == 0)
      if 'eval_jhmdb' in mode:
        pix_pts = viz['tracks']
        grid_size = np.array([
            inputs[input_key]['im_size'][1],
            inputs[input_key]['im_size'][0],
        ])
        pix_pts = transforms.convert_grid_coordinates(
            pix_pts,
            (tapnet_model.TRAIN_SIZE[2], tapnet_model.TRAIN_SIZE[1]),
            grid_size,
        )
        mean_scalars = self._eval_jhmdb(
            pix_pts,
            inputs[input_key]['gt_pose'],
            inputs[input_key]['gt_pose_orig'],
            inputs[input_key]['im_size'],
            inputs[input_key]['fname'],
            is_first=batch_id == 0,
        )
        scalars = {}
      if write_viz:
        pix_pts = viz['tracks']
        targ_pts = None
        if 'eval_kinetics_points' in mode:
          targ_pts = inputs[input_key]['target_points']
        outname = [
            f'{outdir}/{x}.mp4'
            for x in range(batch_size * batch_id, batch_size * (batch_id + 1))
        ]
        write_visualization(
            (inputs[input_key]['video'] + 1.0) * (255.0 / 2.0),
            pix_pts,
            jax.nn.sigmoid(viz['occlusion']),
            outname,
            gt_points=targ_pts,
            gt_occluded=inputs[input_key]['occluded'],
            trackgroup=inputs[input_key]['trackgroup']
            if 'trackgroup' in inputs[input_key]
            else None,
        )
      del viz

      batch_id += 1
      logging.info('eval batch: %d', batch_id)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_map(jnp.add, summed_scalars, scalars)

      if 'eval_jhmdb' not in mode:
        mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
      logging.info(mean_scalars)
    logging.info(evaluation_datasets.latex_table(mean_scalars))

    return mean_scalars

  def _eval_inference(
      self,
      global_step: chex.Array,
      state: chex.ArrayTree,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      wrapped_forward_fn: task.WrappedForwardFn,
  ) -> Mapping[str, chex.Array]:
    """Inferences a single video."""

    def _sample_random_points(frame_max_idx, height, width, num_points):
      """Sample random points with (time, height, width) order."""
      y = np.random.randint(0, height, (num_points, 1))
      x = np.random.randint(0, width, (num_points, 1))
      t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
      points = np.concatenate((t, y, x), axis=-1).astype(np.int32)
      return points

    config = self.config.inference
    input_video_path = config.input_video_path
    output_video_path = config.output_video_path
    resize_height, resize_width = config.resize_height, config.resize_width
    num_points = config.num_points

    logging.info('load video from %s', input_video_path)
    video = media.read_video(input_video_path)
    num_frames, fps = video.metadata.num_images, video.metadata.fps
    logging.info('resize video to (%s, %s)', resize_height, resize_width)
    video = media.resize_video(video, (resize_height, resize_width))
    video = video.astype(np.float32) / 255 * 2 - 1
    query_points = _sample_random_points(
        num_frames, resize_height, resize_width, num_points
    )
    occluded = np.zeros((num_points, num_frames), dtype=np.float32)
    inputs = {
        self.input_key: {
            'video': video[np.newaxis],
            'query_points': query_points[np.newaxis],
            'occluded': occluded[np.newaxis],
        }
    }

    occlusion_logits, tracks, _ = self._infer_batch(
        params,
        state,
        inputs,
        rng,
        wrapped_forward_fn,
        self.input_key,
        query_chunk_size=16,
    )
    occluded = occlusion_logits > 0

    video = (video + 1) * 255 / 2
    video = video.astype(np.uint8)

    painted_frames = viz_utils.paint_point_track(
        video,
        tracks[0],
        ~occluded[0],
    )
    media.write_video(output_video_path, painted_frames, fps=fps)
    logging.info('Inference result saved to %s', output_video_path)

    return {'': 0}

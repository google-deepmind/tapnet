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

"""Perceiver task module."""

import functools
from os import path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jaxline import utils
import matplotlib
import mediapy as media
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

from tapnet import evaluation_datasets
from tapnet import task
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils

matplotlib.use('Agg')


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
      model_key: str = 'tapnet_model',
      prediction_algo: str = 'cost_volume_regressor',
      softmax_temperature: float = 20.0,
      contrastive_loss_weight: float = 0.05,
      position_loss_weight: float = 0.05,
      expected_dist_thresh: float = 6.0,
      train_chunk_size: int = 32,
      eval_chunk_size: int = 16,
      eval_inference_resolution=(256, 256),
      eval_metrics_resolution=(256, 256),
  ):
    """Constructs a task for supervised learning on Kubric.

    Args:
      config: a ConfigDict for configuring this experiment, notably including
        the paths for checkpoints and datasets.
      input_key: The forward pass takes an input dict.  Inference or learning
        will be performed on inputs[input_key]['video']
      model_key: The model to use from shared_modules
      prediction_algo: specifies the network architecture to use to make
        predictions.  Can be 'cost_volume_regressor' for the algorithm presented
        in the TAPNet paper, or 'cost_volume_cycle_consistency' for the VFS-Like
        algorithm presented in the earlier Kubric paper.
      softmax_temperature: temperature applied to cost volume before softmax.
        This is only used with cost_volume_cycle_consistency.
      contrastive_loss_weight: Weight for the additional contrastive loss that's
        applied alongside the trajectory prediction loss.
      position_loss_weight: Weight for position loss.
      expected_dist_thresh: threshold for expected distance. Will be ignored if
        the model does not return expected_dist.
      train_chunk_size: Compute predictions on this many queries simultaneously.
        This saves memory as the cost volumes can be very large.
      eval_chunk_size: Compute predictions on this many queries simultaneously.
        This saves memory as the cost volumes can be very large.
      eval_inference_resolution: The video resolution during model inference in
        (height, width). It can be different from the training resolution.
      eval_metrics_resolution: The video resolution during evaluation metric
        computation in (height, width). Point tracks will be re-scaled to this
        resolution.
    """

    super().__init__()
    self.config = config
    self.input_key = input_key
    self.model_key = model_key
    self.prediction_algo = prediction_algo
    self.contrastive_loss_weight = contrastive_loss_weight
    self.softmax_temperature = softmax_temperature
    self.position_loss_weight = position_loss_weight
    self.expected_dist_thresh = expected_dist_thresh
    self.train_chunk_size = train_chunk_size
    self.eval_chunk_size = eval_chunk_size
    self.eval_inference_resolution = eval_inference_resolution
    self.eval_metrics_resolution = eval_metrics_resolution

  def forward_fn(
      self,
      inputs: chex.ArrayTree,
      is_training: bool,
      shared_modules: Optional[Mapping[str, task.SharedModule]] = None,
      input_key: Optional[str] = None,
      query_chunk_size: int = 32,
      get_query_feats: bool = False,
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
        shared_modules['tapnet_model'] should be a JointModel.
      input_key: Run on inputs[input_key]['video']. If None, use the input_key
        from the constructor.
      query_chunk_size: Compute predictions on this many queries simultaneously.
        This saves memory as the cost volumes can be very large.
      get_query_feats: If True, also return features for each query.

    Returns:
      Result dict produced by calling the joint model. See tapnet_model.py.
    """
    if input_key is None:
      input_key = self.input_key
    frames = inputs[input_key]['video']

    if self.prediction_algo in [
        'cost_volume_regressor',
        'cost_volume_cycle_consistency',
    ]:
      return shared_modules[self.model_key](
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

    output, state = functools.partial(
        wrapped_forward_fn,
        input_key=input_key,
        query_chunk_size=self.train_chunk_size,
    )(params, state, rng, inputs, is_training=is_training)

    def tapnet_loss(
        points, occlusion, target_points, target_occ, shape, expected_dist=None
    ):
      # Huber loss is by default measured under 256x256 resolution
      points = transforms.convert_grid_coordinates(
          points, shape[3:1:-1], (256, 256), coordinate_format='xy'
      )
      target_points = transforms.convert_grid_coordinates(
          target_points, shape[3:1:-1], (256, 256), coordinate_format='xy'
      )
      loss_huber = model_utils.huber_loss(points, target_points, target_occ)
      loss_huber = jnp.mean(loss_huber) * self.position_loss_weight

      if expected_dist is None:
        loss_prob = 0.0
      else:
        loss_prob = model_utils.prob_loss(
            jax.lax.stop_gradient(points),
            expected_dist,
            target_points,
            target_occ,
            self.expected_dist_thresh,
        )
        loss_prob = jnp.mean(loss_prob)

      target_occ = target_occ.astype(occlusion.dtype)  # pytype: disable=attribute-error
      loss_occ = optax.sigmoid_binary_cross_entropy(occlusion, target_occ)
      loss_occ = jnp.mean(loss_occ)
      return loss_huber, loss_occ, loss_prob

    loss_scalars = {}
    loss = 0.0
    if self.prediction_algo in ['cost_volume_regressor']:
      loss_huber, loss_occ, loss_prob = tapnet_loss(
          output['tracks'],
          output['occlusion'],
          inputs[input_key]['target_points'],
          inputs[input_key]['occluded'],
          inputs[input_key]['video'].shape,  # pytype: disable=attribute-error  # numpy-scalars
          output['expected_dist'] if 'expected_dist' in output else None,
      )
      loss = loss_huber + loss_occ + loss_prob
      loss_scalars['position_loss'] = loss_huber
      loss_scalars['occlusion_loss'] = loss_occ
      if 'expected_dist' in output:
        loss_scalars['prob_loss'] = loss_prob

      if 'unrefined_tracks' in output:
        for l in range(len(output['unrefined_tracks'])):
          loss_huber, loss_occ, loss_prob = tapnet_loss(
              output['unrefined_tracks'][l],
              output['unrefined_occlusion'][l],
              inputs[input_key]['target_points'],
              inputs[input_key]['occluded'],
              inputs[input_key]['video'].shape,  # pytype: disable=attribute-error  # numpy-scalars
              output['unrefined_expected_dist'][l]
              if 'unrefined_expected_dist' in output
              else None,
          )
          loss += loss_huber + loss_occ + loss_prob
          loss_scalars[f'position_loss_{l}'] = loss_huber
          loss_scalars[f'occlusion_loss_{l}'] = loss_occ
          if 'unrefined_expected_dist' in output:
            loss_scalars[f'prob_loss_{l}'] = loss_prob

    if self.prediction_algo in ['cost_volume_cycle_consistency']:
      feature_grid = output['feature_grid']
      query_feats = output['query_feats']

      loss_contrast = []

      # This computes the contrastive loss from the paper.  We break the set of
      # queries up into chunks in order to save memory.
      for qchunk in range(0, query_feats.shape[1], self.train_chunk_size):
        qchunk_low = qchunk
        qchunk_high = qchunk + self.train_chunk_size
        all_pairs_dots = jnp.einsum(
            'bnc,bthwc->bnthw',
            query_feats[:, qchunk_low:qchunk_high],
            feature_grid,
        )
        all_pairs_softmax = jax.nn.log_softmax(
            all_pairs_dots * self.softmax_temperature, axis=(2, 3, 4)
        )
        im_shp = inputs[input_key]['video'].shape  # pytype: disable=attribute-error  # numpy-scalars
        target_points = inputs[input_key]['target_points']
        position_in_grid2 = transforms.convert_grid_coordinates(
            target_points[:, qchunk_low:qchunk_high],
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
        interp_softmax = jax.vmap(jax.vmap(jax.vmap(model_utils.interp)))(
            all_pairs_softmax,
            position_in_grid2[..., ::-1],
        )

        target_occ = inputs[input_key]['occluded']
        target_occ = target_occ[:, qchunk_low:qchunk_high]
        loss_contrast.append(
            jnp.mean(interp_softmax * (1.0 - target_occ), axis=-1)
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
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    Returns:
      A 2-tuple consisting of the occlusion logits, of shape
        [batch, num_queries, num_frames], the predicted position, of shape
        [batch, num_queries, num_frames, 2], and a dict of loss scalars.
    """
    # Features for each query point are required when using cycle consistency.
    get_query_feats = self.prediction_algo in ['cost_volume_cycle_consistency']
    output, _ = functools.partial(wrapped_forward_fn, input_key=input_key)(
        params,
        state,
        rng,
        inputs,
        is_training=False,
        query_chunk_size=self.eval_chunk_size,
        get_query_feats=get_query_feats,
    )
    loss_scalars = {}
    if self.prediction_algo in ['cost_volume_regressor']:
      # Outputs are already in the correct format for cost_volume_regressor.
      tracks = output['tracks']
      loss_occ = optax.sigmoid_binary_cross_entropy(
          output['occlusion'], inputs[input_key]['occluded']
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
      for qchunk in range(0, query_feats.shape[1], self.eval_chunk_size):
        # Compute pairwise dot products between queries and all other features
        all_pairs_dots = jnp.einsum(
            'bnc,bthwc->bnthw',
            query_feats[:, qchunk : qchunk + self.eval_chunk_size],
            feature_grid,
        )
        # Compute the soft argmax for each frame
        query_point_chunk = inputs[input_key]['query_points'][
            :, qchunk : qchunk + self.eval_chunk_size
        ]
        im_shp = inputs[input_key]['video'].shape  # pytype: disable=attribute-error  # numpy-scalars
        tracks = model_utils.heatmaps_to_points(
            jax.nn.softmax(
                all_pairs_dots * self.softmax_temperature, axis=(-1, -2)
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
            [frame_id, tracks[..., ::-1]], axis=-1
        )
        position_in_grid = transforms.convert_grid_coordinates(
            position_in_grid,
            im_shp[1:4],
            feature_grid.shape[1:4],
            coordinate_format='tyx',
        )
        # vmap over channels, duplicating the coordinates
        vmap_interp = jax.vmap(
            model_utils.interp, in_axes=(3, None), out_axes=1
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
        query_frame = transforms.convert_grid_coordinates(
            inputs[input_key]['query_points'][
                :, qchunk : qchunk + self.eval_chunk_size, ...
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
            'bntc,bnhwc->bnthw', interp_features, target_features
        )

        # Again, take the soft argmax to see if we come back to the place we
        # started from.
        # inverse_tracks is [batch_size, chunk, num_frames, 2]
        inverse_tracks = model_utils.heatmaps_to_points(
            jax.nn.softmax(
                all_pairs_dots * self.softmax_temperature,
                axis=(-2, -1),
            ),
            im_shp,
        )
        query_points = inputs[input_key]['query_points']
        query_points = query_points[:, qchunk : qchunk + self.eval_chunk_size]
        dist = jnp.square(inverse_tracks - query_points[jnp.newaxis, 2:0:-1])
        dist = jnp.sum(dist, axis=-1)
        occlusion = dist > jnp.square(48.0)
        # We need to return logits, but the cycle consistency rule is binary.
        # So we just convert the binary values into large real values.
        occlusion = occlusion * 20.0 - 10.0

        all_occlusion.append(occlusion)
        all_tracks.append(tracks)

      tracks = jnp.concatenate(all_tracks, axis=1)
      occlusion = jnp.concatenate(all_occlusion, axis=1)

    outputs = {'tracks': tracks, 'occlusion': occlusion}
    if 'expected_dist' in output:
      outputs['expected_dist'] = output['expected_dist']
    return outputs, loss_scalars

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
    outputs, loss_scalars = self._infer_batch(
        params, state, inputs, rng, wrapped_forward_fn, input_key
    )
    loss_scalars = {**loss_scalars}  # Mutable copy.

    gt_occluded = inputs[input_key]['occluded']
    gt_target_points = inputs[input_key]['target_points']
    query_points = inputs[input_key]['query_points']

    tracks = outputs['tracks']
    # Huber loss is by default measured under 256x256 resolution
    shape = inputs[input_key]['video'].shape
    target_points = transforms.convert_grid_coordinates(
        gt_target_points, shape[3:1:-1], (256, 256), coordinate_format='xy'
    )
    points = transforms.convert_grid_coordinates(
        tracks, shape[3:1:-1], (256, 256), coordinate_format='xy'
    )
    loss_huber = model_utils.huber_loss(points, target_points, gt_occluded)
    loss_scalars['position_loss'] = loss_huber

    occlusion_logits = outputs['occlusion']
    pred_occ = jax.nn.sigmoid(occlusion_logits)
    if 'expected_dist' in outputs:
      expected_dist = outputs['expected_dist']
      pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
    pred_occ = pred_occ > 0.5  # threshold

    if self.eval_inference_resolution != self.eval_metrics_resolution:
      # Resize prediction and groundtruth to standard evaluation resolution
      query_points = transforms.convert_grid_coordinates(
          query_points,
          (1,) + inputs[input_key]['video'].shape[2:4],  # (1, height, width)
          (1,) + self.eval_metrics_resolution,  # (1, height, width)
          coordinate_format='tyx',
      )
      gt_target_points = transforms.convert_grid_coordinates(
          gt_target_points,
          inputs[input_key]['video'].shape[3:1:-1],  # (width, height)
          self.eval_metrics_resolution[::-1],  # (width, height)
          coordinate_format='xy',
      )
      tracks = transforms.convert_grid_coordinates(
          tracks,
          inputs[input_key]['video'].shape[3:1:-1],  # (width, height)
          self.eval_metrics_resolution[::-1],  # (width, height)
          coordinate_format='xy',
      )

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
      yield from evaluation_datasets.create_kubric_eval_train_dataset(
          mode, train_size=self.config.datasets.kubric_kwargs.train_size
      )
    elif 'eval_kubric' in mode:
      yield from evaluation_datasets.create_kubric_eval_dataset(
          mode, train_size=self.config.datasets.kubric_kwargs.train_size
      )
    elif 'eval_davis_points' in mode:
      yield from evaluation_datasets.create_davis_dataset(
          self.config.davis_points_path,
          query_mode=query_mode,
          resolution=self.eval_inference_resolution,
      )
    elif 'eval_jhmdb' in mode:
      yield from evaluation_datasets.create_jhmdb_dataset(
          self.config.jhmdb_path, resolution=self.eval_inference_resolution
      )
    elif 'eval_robotics_points' in mode:
      yield from evaluation_datasets.create_rgb_stacking_dataset(
          self.config.robotics_points_path,
          query_mode=query_mode,
          resolution=self.eval_inference_resolution,
      )
    elif 'eval_kinetics_points' in mode:
      yield from evaluation_datasets.create_kinetics_dataset(
          self.config.kinetics_points_path,
          query_mode=query_mode,
          resolution=self.eval_inference_resolution,
      )
    elif 'eval_robotap' in mode:
      yield from evaluation_datasets.create_csv_dataset(
          dataset_name='robotap',
          csv_path=self.config.robotap_csv_path,
          video_base_path=self.config.robotap_video_path,
          query_mode=query_mode,
          resolution=self.eval_inference_resolution,
      )
    elif 'eval_perception_test' in mode:
      yield from evaluation_datasets.create_csv_dataset(
          dataset_name='perception_test',
          csv_path=self.config.perception_test_csv_path,
          video_base_path=self.config.perception_test_video_path,
          query_mode=query_mode,
          resolution=self.eval_inference_resolution,
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
          gt_poses[:, 0:1, 0] < 0, gt_poses[:, 0:1, 0] >= width
      )
      invalid_y = np.logical_or(
          gt_poses[:, 0:1, 1] < 0, gt_poses[:, 0:1, 1] >= height
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

    outdir = path.join(self.config.checkpoint_dir, mode, str(global_step))
    logging.info('Saving videos to %s', outdir)

    try:
      tf.io.gfile.makedirs(outdir)
    except FileExistsError:
      print(f'Path {outdir} exists. Skip creating a new dir.')

    if 'eval_kinetics' in mode:
      input_key = 'kinetics'
    elif 'eval_davis_points' in mode:
      input_key = 'davis'
    elif 'eval_jhmdb' in mode:
      input_key = 'jhmdb'
    elif 'eval_robotics_points' in mode:
      input_key = 'robotics'
    elif 'eval_robotap' in mode:
      input_key = 'robotap'
    elif 'eval_perception_test' in mode:
      input_key = 'perception_test'
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
      write_viz = batch_id < 0
      if 'eval_davis_points' in mode or 'eval_robotics_points' in mode:
        # Only write videos sometimes for the small datasets; otherwise
        # there will be a crazy number of videos dumped.
        write_viz = write_viz and (global_step % 10 == 0)
      if 'eval_jhmdb' in mode:
        pix_pts = viz['tracks']
        grid_size = np.array(
            [inputs[input_key]['im_size'][1], inputs[input_key]['im_size'][0]]
        )
        pix_pts = transforms.convert_grid_coordinates(
            pix_pts,
            (
                self.eval_inference_resolution[1],
                self.eval_inference_resolution[0],
            ),
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
        if 'eval_kinetics' in mode:
          targ_pts = inputs[input_key]['target_points']
        outname = [
            f'{outdir}/{x}.mp4'
            for x in range(batch_size * batch_id, batch_size * (batch_id + 1))
        ]
        viz_utils.write_visualization(
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

    outputs, _ = self._infer_batch(
        params,
        state,
        inputs,
        rng,
        wrapped_forward_fn,
        self.input_key,
    )
    occluded = outputs['occlusion'] > 0

    video = (video + 1) * 255 / 2
    video = video.astype(np.uint8)

    painted_frames = viz_utils.paint_point_track(
        video,
        outputs['tracks'][0],
        ~occluded[0],
    )
    media.write_video(output_video_path, painted_frames, fps=fps)
    logging.info('Inference result saved to %s', output_video_path)

    return {'': 0}  # pytype: disable=bad-return-type  # numpy-scalars

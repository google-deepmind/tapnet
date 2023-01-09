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

"""Abstract task interface with documentation.

"""

import abc
from typing import Mapping, Optional, Tuple

import chex
import typing_extensions


class SharedModule(typing_extensions.Protocol):

  def __call__(
      self,
      video: chex.Array,
      is_training: bool,
      query_points: chex.Array,
      query_chunk_size: Optional[int] = None,
      get_query_feats: bool = False,
      **kwargs,
  ) -> Mapping[str, chex.Array]:
    """Runs a forward pass of a module.

    Args:
      video: A 4-D or 5-D tensor representing a batch of sequences of images. In
        the 4-D case, we assume the entire batch has been concatenated along the
        batch dimension, one sequence after the other.  This can speed up
        inference on the TPU and save memory.
      is_training: Whether we are training.
      query_points: The query points for which we compute tracks.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      get_query_feats: If True, also return the features for each query obtained
        using bilinear interpolation from the feature grid
      **kwargs: Additional module-specific parameters.

    Returns:
      Module outputs.
    """


class WrappedForwardFn(typing_extensions.Protocol):
  """Forward function, wrapped by haiku.

  This wrapped forward function will inject the shared_modules and allow them
  to use shared params. It should be called inside a loss_fn using the same
  signature as `Task.forward_fn` (minus the shared_modules).
  """

  def __call__(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      rng: chex.PRNGKey,
      inputs: chex.ArrayTree,
      is_training: bool,
      input_key: Optional[str] = None,
      query_chunk_size: int = 16,
      get_query_feats: bool = True,
  ) -> Mapping[str, chex.Array]:
    """Forward pass for predicting point tracks.

    Args:
      params: hk.Params with the model parameters
      state: hk.State with the model state
      rng: jax.random.PRNGKey for random number generation.
      inputs: Input dict. Inference will be performed on will be performed on
        inputs[input_key]['video'] (with fallback to the input_key specified in
        the constructor). Input videos should be a standard video tensor
        ([batch, num_frames, height, width, 3]) normalize to [-1,1].
        inputs[input_key]['query_points'] specifies the query point locations,
        of shape [batch, num_queries, 3], where each query is [t,y,x]
        coordinates normalized to between -1 and 1.
      is_training: Whether the model is in training mode.
      input_key: Run on inputs[input_key]['video']. If None, use the input_key
        from the constructor.
      query_chunk_size: Compute predictions on this many queries simultaneously.
        This saves memory as the cost volumes can be very large.
      get_query_feats: If True, also return features for each query.

    Returns:
      Result dict produced by calling the model.
    """


class Task(abc.ABC):
  """An abstract Task definition."""

  @abc.abstractmethod
  def forward_fn(
      self,
      inputs: chex.ArrayTree,
      is_training: bool,
      shared_modules: Optional[Mapping[str, SharedModule]] = None,
  ) -> chex.ArrayTree:
    """Run the model forward pass and construct all required Haiku modules.

    Args:
      inputs: model input tensors. This is a dict keyed by dataset name, where
        the value for each key is an item from the specified dataset.
      is_training: Is the forward pass in training mode or not.
      shared_modules: A dict of Haiku modules, keyed by module name, which
        can be used to construct the modules which are shared across different
        tasks.

    Returns:
      Anything. The important part is that this must construct all modules that
        Haiku needs to initialize.

    """

  def get_gradients(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      global_step: chex.Array,
      wrapped_forward_fn: WrappedForwardFn,
      is_training: bool = True,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, Mapping[str, chex.Array]]:
    """Get gradients for this tasks's loss function.

    Params, state, inputs, rng, and global_step are pmapped, i.e. a separate
      copy on each device.

    Args:
      params: Haiku parameters
      state: Haiku state
      inputs: model input tensors. This is a dict keyed by dataset name, where
        the value for each key is an item from the specified dataset.
      rng: random number state
      global_step: global step
      wrapped_forward_fn: A wrapper for the forward function that will inject
        the shared_modules and allow them to use shared params. It should be
        called inside a loss_fn using the same signature as forward_fn
        (minus the shared_modules).
      is_training: Is the forward pass in training mode or not.

    Returns:
      grads: A set of gradients compatible with optax apply_gradients (these
        will be summed across tasks).
      state: An updated Haiku state. The returned state will be passed to the
        next task in the list.
      scalars: A dict of (pmapped) scalars to be logged for this task. All
        dict keys will have the task name prepended before they are logged.

    """
    raise NotImplementedError()

  def evaluate(
      self,
      global_step: chex.Array,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      rng: chex.PRNGKey,
      wrapped_forward_fn: WrappedForwardFn,
      mode: str,
  ) -> Mapping[str, chex.Array]:
    """Evaluate this task's performance on a downstream benchmark.

    Args:
      global_step: global step
      params: Haiku parameters
      state: Haiku state
      rng: random number state
      wrapped_forward_fn: A wrapper for the forward function that will inject
        the shared_modules and allow them to use shared params.
      mode: A string mode used to determine, e.g., which dataset or split to
        evaluate on. This will be the same value as the 'mode' parameter
        used to launch different eval jobs in Jaxline.

    Returns:
      scalars: A dict of scalars to be logged for this task. All
        dict keys will have the task name prepended before they are logged.

    """
    raise NotImplementedError()




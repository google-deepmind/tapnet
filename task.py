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

"""Abstract task interface with documentation.

"""

import abc


class Task(abc.ABC):
  """An abstract Task definition."""

  @abc.abstractmethod
  def forward_fn(
      self,
      inputs,
      is_training,
      shared_modules=None,
  ):
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
      params,
      state,
      inputs,
      rng,
      global_step,
      wrapped_forward_fn,
      is_training=True,
  ):
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
    del params
    del inputs
    del rng
    del global_step
    del wrapped_forward_fn
    del is_training
    return None, state, {}

  def evaluate(
      self,
      global_step,
      params,
      state,
      rng,
      wrapped_forward_fn,
      mode
  ):
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
    del global_step
    del params
    del state
    del rng
    del wrapped_forward_fn
    del mode
    return {}




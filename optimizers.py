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

"""Optimizer utils."""
from typing import Callable, Sequence, NamedTuple, Optional, Text

import haiku as hk
import jax
import jax.numpy as jnp
import optax

NORM_NAMES = ["layer_norm", "batch_norm", "_bn", "linear_classifier"]


def _weight_decay_exclude(
    exclude_names: Optional[Sequence[Text]] = None,
) -> Callable[[str, str, jnp.ndarray], bool]:
  """Logic for deciding which parameters to include for weight decay..

  Args:
    exclude_names: an optional list of names to include for weight_decay. ['w']
      by default.

  Returns:
    A predicate that returns False for params that need to be excluded from
    weight_decay.
  """
  # By default weight_decay the weights but not the biases.
  if exclude_names is None:
    exclude_names = ["b"]

  def include(module_name: Text, name: Text, value: jnp.ndarray):
    del value
    # Do not weight decay the parameters of normalization blocks.
    if any([norm_name in module_name for norm_name in NORM_NAMES]):
      return False
    else:
      return name not in exclude_names

  return include


class AddWeightDecayState(NamedTuple):
  """Stateless transformation."""


def add_weight_decay(
    weight_decay: float,
    exclude_names: Optional[Sequence[Text]] = None,
) -> optax.GradientTransformation:
  """Add parameter scaled by `weight_decay` to the `updates`.

  Same as optax.add_decayed_weights but can exclude some parameters.

  Args:
    weight_decay: weight_decay coefficient.
    exclude_names: an optional list of names to exclude for weight_decay. ['b']
      by default.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return AddWeightDecayState()

  def update_fn(updates, state, params):
    include = _weight_decay_exclude(exclude_names=exclude_names)

    u_in, u_ex = hk.data_structures.partition(include, updates)
    p_in, _ = hk.data_structures.partition(include, params)
    u_in = jax.tree_map(lambda g, p: g + weight_decay * p, u_in, p_in)
    updates = hk.data_structures.merge(u_ex, u_in)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)

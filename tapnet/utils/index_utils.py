# Copyright 2025 Google LLC
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

"""Index utils for TAPViT-SSM."""

import functools

import chex
import jax
import jax.numpy as jnp
# from kauldron.typing import Int, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member


def scatter_inner(
    target: chex.Array, mask: chex.Array, timestep: chex.Array, data: chex.Array
) -> chex.Array:
  """Scatter data into target at timestep if mask is true.

  Args:
    target: (T, c) float target tensor
    mask: ([]) bool mask
    timestep: ([]) int timestep
    data: (c,) data to scatter
  Returns:
    (T, c) updated target tensor
  """
  updated_target = target.at[timestep].set(data)
  return jnp.where(mask, updated_target, target)


@jax.vmap
@functools.partial(jax.vmap, in_axes=(1, 0, 0, 0), out_axes=1)
def scatter(
    target: chex.Array, mask: chex.Array, timestep: chex.Array, data: chex.Array
) -> chex.Array:
  """Scatter data into target at timestep if mask is true.

  (dimensions that are added via vmap are put into square brackets [])

  Args:
    target: ([B], T, [Q], c) float target tensor
    mask: ([B, Q]) bool mask
    timestep: ([B, Q]) int timestep
    data: ([B, Q], c,) data to scatter
  Returns:
    ([B], T, [Q], c) updated target tensor
  """
  return scatter_inner(target, mask, timestep, data)


@jax.vmap
@functools.partial(jax.vmap, in_axes=(1, None, None, 0), out_axes=1)
def scatter2(
    target: chex.Array, mask: chex.Array, timestep: chex.Array, data: chex.Array
) -> chex.Array:
  """Scatter data into target at timestep if mask is true.

  (dimensions that are added via vmap are put into square brackets [])

  Args:
    target: ([B], T, [N], c) float target tensor
    mask: ([B]) bool mask
    timestep: ([B]) int timestep
    data: ([B, N], c,) data to scatter
  Returns:
    ([B], T, [N], c) updated target tensor
  """
  return scatter_inner(target, mask, timestep, data)


@jax.vmap
@functools.partial(jax.vmap, in_axes=(1, 0, 0, 0), out_axes=1)
def scatter_prefix(
    target: chex.Array, mask: chex.Array, timestep: chex.Array, data: chex.Array
) -> chex.Array:
  """Scatter data into target before timestep if mask is true.

  Equivalent to

  updated_target = target.at[:timestep].set(data)
  return jnp.where(mask, updated_target, target)

  but works in a static way.

  (dimensions that are added via vmap are put into square brackets [])

  Args:
    target: ([B], T, [Q], c) float target tensor
    mask: ([B, Q]) bool mask
    timestep: ([B, Q]) int timestep
    data: ([B, Q], c,) data to scatter
  Returns:
    ([B], T, [Q], c) updated target tensor
  """
  cond = (jnp.arange(target.shape[0]) < timestep) & mask
  return jnp.where(
      jnp.tile(cond[:, None], (1, target.shape[1])),
      jnp.tile(data, (target.shape[0], 1)),
      target,
  )


@jax.vmap
@functools.partial(jax.vmap, in_axes=(1, 0, 0, 0), out_axes=1)
def scatter_suffix(
    target: chex.Array, mask: chex.Array, timestep: chex.Array, data: chex.Array
) -> chex.Array:
  """Scatter data into target before timestep if mask is true.

  Equivalent to

  updated_target = target.at[timestep:].set(data)
  return jnp.where(mask, updated_target, target)

  but works in a static way.

  (dimensions that are added via vmap are put into square brackets [])

  Args:
    target: ([B], T, [Q], c) float target tensor
    mask: ([B, Q]) bool mask
    timestep: ([B, Q]) int timestep
    data: ([B, Q], c,) data to scatter
  Returns:
    ([B], T, [Q], c) updated target tensor
  """
  cond = (jnp.arange(target.shape[0]) >= timestep) & mask
  return jnp.where(
      jnp.tile(cond[:, None], (1, target.shape[1])),
      jnp.tile(data, (target.shape[0], 1)),
      target,
  )

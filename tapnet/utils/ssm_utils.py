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

"""Util functions for processing videos with SSMs."""

import enum
import chex
import einops
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from recurrentgemma._src.jax import scan

from tapnet.utils import index_utils


class CodeOrigin(enum.StrEnum):
  XLM = "xlm"
  THIRD_PARTY = "third_party"


# @typechecked
def transpose_flatten(
    x: chex.Array, like_shape: tuple[int, int, int, int], original_shape: str
) -> chex.Array:
  shape = dict(zip("btnc", like_shape, strict=True))
  return einops.rearrange(x, original_shape + "-> (b n) t c", **shape)


# @typechecked
def unflatten_untranspose(
    x: chex.Array, like_shape: tuple[int, int, int, int], original_shape: str
) -> chex.Array:
  shape = dict(zip("btnc", like_shape, strict=True))
  return einops.rearrange(x, "(b n) t c ->" + original_shape, **shape)


def get_sharding_spec() -> scan.ShardingSpec:
  """Returns the sharding spec for the Pallas kernel."""
  devices = np.asarray(jax.devices())
  grid = devices.reshape(-1, len(devices))

  # The axis names used in Kauldron are 'i' and 'j'
  mesh = jax.sharding.Mesh(grid, axis_names=("i", "j"))
  # It looks like `j` is the data axis
  scan_sharding_spec = scan.ShardingSpec(
      mesh=mesh,
      batch_axis_name="j",
      activations_axis_name="i",
  )
  return scan_sharding_spec


class TokenSubsampling(nn.Module):
  """Drops video tubes."""

  # Hparams.
  drop_ratio: float
  drop_ratio_test: float = 0.0
  shuffle_tokens: bool = True

  # only true is supported for now
  mask_temporal_tokens: bool = True

  is_training: bool = False

  @nn.compact
  # @typechecked
  def __call__(
      self,
      tokens: chex.Array,  # Float["*B T N D"],
      mask_token: chex.Array,  # Float["*B T N D"],
      override_drop_ratio: float | None = None,
  ) -> tuple[chex.Array, chex.Array]:  # Float["*B T N D"], Bool["*B T"]]:
    """Drops tokens randomly."""
    n_batch, seq_len, num_tokens, _ = tokens.shape

    # By default tokens are only dropped for training.
    if override_drop_ratio is not None:
      drop_ratio = override_drop_ratio
    elif self.is_training:
      drop_ratio = self.drop_ratio
    else:
      drop_ratio = self.drop_ratio_test
    if drop_ratio == 0.0:
      return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_)

    # Drop tokens randomly.
    if self.mask_temporal_tokens:
      n_tokens = int(seq_len) - 1
    else:
      n_tokens = int(num_tokens)
    if len(n_batch) != 1:
      raise NotImplementedError("*B is not supported yet!")
    rng = self.make_rng("degradation")
    num_vis_patches = tokens.shape[2]

    subkey, _ = jax.random.split(rng, 2)

    masked_tokens = tokens

    subsample_size = jax.random.choice(subkey, n_tokens - 1, shape=(n_batch,))
    subsample_size += 1
    # subsample_size is a random integer between 1 and T - 1 (inclusively)

    # mask_tokens - B T N D
    mask = jnp.ones(
        (n_batch, num_vis_patches), dtype=jnp.bool_)
    indices = jnp.tile(
        subsample_size[:, None], (1, num_vis_patches)
    )
    # TODO(zholus): this works because we don't have temporal positional
    # embeddings (i.e. at all temporal positions masked tokens are the same).
    # when we have positional embeddings, we need dynamically choose
    # the mask token for each position according to `subsample_size`.
    scatter_data = mask_token[:, 0]
    # scatter_data.shape == [B, N, D]
    masked_tokens = index_utils.scatter_suffix(
        masked_tokens, mask, indices, scatter_data
    )
    masked_positions = jnp.zeros(
        (n_batch, n_tokens + 1, 1, 1), dtype=jnp.bool_)
    mask = jnp.ones(
        (n_batch, 1), dtype=jnp.bool_)
    masked_positions = index_utils.scatter_suffix(
        masked_positions, mask, subsample_size[:, None],
        jnp.ones((n_batch, 1, 1), dtype=jnp.bool_)
    )[..., 0, 0]
    return masked_tokens, masked_positions

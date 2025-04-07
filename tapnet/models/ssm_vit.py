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

"""Masked sequence decoder model.

Uses SSM as temporal model and ViT for spatial attention.
"""

import functools
from typing import Optional, Sequence, Union

import chex
import einops
import flax
import flax.linen as nn
from gemma import modules as gemma_modules
import jax
import jax.numpy as jnp
import numpy as np
from recurrentgemma._src import common
from recurrentgemma._src.jax import array_typing as at
from recurrentgemma._src.jax import modules as recurrentgemma_modules

from tapnet.utils import index_utils
from tapnet.utils import model_utils
from tapnet.utils import ssm_utils


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(
        name,
        nn.initializers.normal(stddev=1 / np.sqrt(width)),
        (1, np.prod(seqshape), width),
        dtype,
    )
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[-1]
    x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
    # In some extreme batch-size cases, this is needed as of Sept 2024:
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, dtype=self.dtype_mm, **inits)(x)
    return x


class ViTBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"
  mask_image2image: bool = False
  mask_query2image: bool = False

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)
    attn_mask = jnp.ones(
        (x.shape[0], self.num_heads, x.shape[1], x.shape[1]),
        dtype=jnp.bool_
    )
    if self.mask_image2image:
      attn_mask = attn_mask.at[:, :, :1024, :1024].set(False)
    if self.mask_query2image:
      attn_mask = attn_mask.at[:, :, :1024, 1024:].set(False)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
    )(y, y, sow_weights=True, mask=attn_mask)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x, out


class ViTSSMBlock(nn.Module):
  """A block that chains a SSM block over time with a ViT block over space.
  """
  depth: int
  width: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"
  lru_width: Optional[int] = None
  attention_window_size: int = 2048
  scan_type: common.ScanType = common.ScanType.LINEAR_NATIVE
  bidirectional_ssm: bool = False
  dtype: at.dtype = jnp.float32
  param_dtype: at.dtype = jnp.float32
  mask_image2image: bool = False
  mask_query2image: bool = False
  attention_ablation: bool = False

  @nn.compact
  def __call__(self, x, cache=None, batch=1, deterministic=True):
    # x shape [b * t, h * w + q, c]
    shape = (batch, x.shape[0] // batch) + x.shape[1:]
    b, t, n, _ = shape
    arange = jnp.arange(t)
    # this is very important - if we have cache, we should not reset the
    # hidden state in the beginning of the block.
    if cache is not None:
      arange += 1
    pos = jnp.repeat(arange[None], b*n, axis=0)
    outs = {}
    if not self.attention_ablation:
      ssm_block = recurrentgemma_modules.ResidualBlock(
          name="ssm_block",
          width=self.width * 2 if self.bidirectional_ssm else self.width,
          mlp_expanded_width=self.mlp_dim,
          num_heads=self.num_heads,
          lru_width=self.lru_width,
          attention_window_size=self.attention_window_size,
          temporal_block_type=common.TemporalBlockType.RECURRENT,
          scan_type=self.scan_type,
          final_w_init_variance_scale=2.0 / self.depth,
          scan_sharding_spec=ssm_utils.get_sharding_spec(),
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )
      temporal_block = ssm_block
    else:
      attn_block = gemma_modules.Block(
          name="attn_block",
          num_heads=self.num_heads,
          num_kv_heads=self.num_heads,
          embed_dim=self.width,
          head_dim=self.width // self.num_heads,
          hidden_dim=self.mlp_dim,
          use_post_attn_norm=False,
          use_post_ffw_norm=False,
          sliding_window_size=self.attention_window_size,
          attn_type=gemma_modules.AttentionType.GLOBAL,
          query_pre_attn_scalar=(self.width // self.num_heads) ** -0.5,
          transpose_gating_einsum=False,
      )
      temporal_block = attn_block
      if cache is not None:
        raise ValueError("Cache is not supported for temporal attention.")
    # x shape [b * t, h * w + q, c]
    x = ssm_utils.transpose_flatten(x, shape, "(b t) n c")
    # x shape [b * (h * w + q), t, c]
    if self.bidirectional_ssm:
      bidirectional_x = jnp.concatenate([x, jnp.flip(x, axis=1)], axis=-1)
      assert isinstance(temporal_block, recurrentgemma_modules.ResidualBlock)
      bidirectional_x, _ = temporal_block(bidirectional_x, pos)
      x_fwd, x_bwd = jnp.split(bidirectional_x, 2, axis=-1)
      x = x_fwd + jnp.flip(x_bwd, axis=1)
    else:
      if self.attention_ablation:
        assert isinstance(temporal_block, gemma_modules.Block)
        attn_mask = jnp.ones((x.shape[0], t, t), dtype=jnp.bool_)
        _, x = temporal_block(x, pos, attn_mask=attn_mask, cache=None)
      else:
        assert isinstance(temporal_block, recurrentgemma_modules.ResidualBlock)
        x, outs["ssm_block_cache"] = temporal_block(x, pos, cache)
    # x shape [b * (h * w + q), t, c]
    x = ssm_utils.unflatten_untranspose(x, shape, "(b t) n c")
    # x shape [b * t, h * w + q, c]
    outs["ssm_block"] = x
    vit_block = ViTBlock(
        name="vit_block",
        dtype_mm=self.dtype_mm,
        mlp_dim=self.mlp_dim, num_heads=self.num_heads,
        mask_image2image=self.mask_image2image,
        mask_query2image=self.mask_query2image,
        dropout=self.dropout)
    x, outs["vit_block_intermediates"] = vit_block(x, deterministic)
    outs["vit_block"] = x
    return x, outs


class ViTSSMBackbone(nn.Module):
  """Video Encoder with Spatial Attention and Temporal SSM."""
  depth: int
  width: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  lru_width: Optional[int] = None
  attention_window_size: int = 2048
  scan_type: common.ScanType = common.ScanType.LINEAR_NATIVE
  bidirectional_ssm: bool = False
  dtype: at.dtype = jnp.float32
  param_dtype: at.dtype = jnp.float32
  code_origin: ssm_utils.CodeOrigin = ssm_utils.CodeOrigin.THIRD_PARTY
  remat: bool = False
  mask_image2image: bool = False
  mask_query2image: bool = False
  attention_ablation: bool = False

  @nn.compact
  def __call__(self, x, cache=None, deterministic=True):
    out = {}
    b, t, n, c = x.shape
    x = jnp.reshape(x, [b * t, n, c])

    # Input Encoder
    if not self.remat:
      scan_out = {"ssm_block_cache": []}
      for lyr in range(self.depth):
        # x shape [b * t, h * w + q, c]
        current_cache = jax.tree.map(lambda o, l=lyr: o[l], cache)
        x, out[f"block{lyr:02d}"] = ViTSSMBlock(
            name=f"encoderblock_{lyr}",
            depth=self.depth,
            width=self.width,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            lru_width=self.lru_width,
            attention_window_size=self.attention_window_size,
            scan_type=self.scan_type,
            bidirectional_ssm=self.bidirectional_ssm,
            dtype=self.dtype,
            mask_image2image=self.mask_image2image,
            mask_query2image=self.mask_query2image,
            param_dtype=self.param_dtype,
        )(x, current_cache, b, deterministic)
        scan_out["ssm_block_cache"].append(
            out[f"block{lyr:02d}"]["ssm_block_cache"])
      scan_out["ssm_block_cache"] = jax.tree.map(
          lambda *args: jnp.stack(args, axis=0), *scan_out["ssm_block_cache"])
    else:
      block = nn.remat(
          ViTSSMBlock,
          prevent_cse=False,
          static_argnums=(3, 4),  # "batch" and "deterministic"
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )
      encoder_class = nn.scan(
          block,
          variable_axes={"params": 0, "intermediates": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=(0, nn.broadcast, nn.broadcast),
          length=self.depth
      )
      encoder = encoder_class(
          name="encoderblock",
          depth=self.depth,
          width=self.width,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          dtype_mm=self.dtype_mm,
          mask_image2image=self.mask_image2image,
          mask_query2image=self.mask_query2image,
          attention_ablation=self.attention_ablation,
          lru_width=self.lru_width,
          attention_window_size=self.attention_window_size,
          scan_type=self.scan_type,
          bidirectional_ssm=self.bidirectional_ssm,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )
      x, scan_out = encoder(x, cache, b, deterministic)
      for lyr in range(self.depth):
        out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)
    out["pre_ln"] = x  # Alias for last block, but without the number in it.
    if not self.attention_ablation:
      out["ssm_block_cache"] = scan_out["ssm_block_cache"]
    return (
        nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x),
        out
    )


@flax.struct.dataclass
class TAPNextTrackingState:
  """State for TAPNext."""
  step: int
  query_points: chex.Array  # Float["*B Q t 3"] | Float["*B Q 3"]
  query_padding: chex.Array  # Float["*B Q t"] | Float["*B Q"]
  hidden_state: Optional[recurrentgemma_modules.RecurrentBlockCache] = None


@flax.struct.dataclass
class TAPNextVideoFeatures:
  step: int
  video_features: chex.Array  # Float["*B L T N C"]


class MaskedSequenceDecoder(nn.Module):
  """Masked sequence decoder model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (1, 16, 16)
  image_size: Sequence[int] = (256, 256)
  width: int = 768
  lru_width: Optional[int] = None
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  posemb_full: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "queries"  # Can also be "map" or "tok"
  bidirectional_ssm: bool = False
  head_zeroinit: bool = True
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  remat: bool = False
  dtype_mm: str = "float32"
  dtype_ssm: str = "float32"
  param_dtype_ssm: str = "float32"
  # whether to have self-attention along the spatiotemporal_attention dimension
  spatiotemporal_attn: bool = False
  code_origin: ssm_utils.CodeOrigin = ssm_utils.CodeOrigin.THIRD_PARTY
  query_scale: int = 1
  mask_image2image: bool = False
  mask_query2image: bool = False
  attention_ablation: bool = False

  def setup(self):
    self.lin_proj = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding="VALID",
        name="embedding",
        dtype=self.dtype_mm,
    )
    self.encoder = ViTSSMBackbone(
        depth=self.depth,
        width=self.width,
        lru_width=self.lru_width,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        bidirectional_ssm=self.bidirectional_ssm,
        mask_image2image=self.mask_image2image,
        mask_query2image=self.mask_query2image,
        attention_ablation=self.attention_ablation,
        remat_policy=self.remat_policy,
        remat=self.remat,
        dtype_mm=self.dtype_mm,
        dtype=jnp.bfloat16 if self.dtype_ssm == "bfloat16" else jnp.float32,
        param_dtype=jnp.bfloat16
        if self.param_dtype_ssm == "bfloat16" else jnp.float32,
        name="Transformer",
    )
    self.mask_token = self.param(
        "mask_token",
        nn.initializers.normal(stddev=1 / np.sqrt(self.width)),
        (1, 1, 1, self.width), jnp.float32)
    self.unknown_token = self.param(
        "unknown_token",
        nn.initializers.normal(stddev=1 / np.sqrt(self.width)),
        (1, 1, self.width), jnp.float32)
    self.point_query_token = self.param(
        "point_query_token",
        nn.initializers.normal(stddev=1 / np.sqrt(self.width)),
        (1, 1, 1, self.width), jnp.float32)
    self.dropout_module = nn.Dropout(rate=self.dropout)
    if self.posemb == "learn":
      h = self.image_size[0] // self.patch_size[1]
      w = self.image_size[1] // self.patch_size[2]
      c = self.width
      self.image_pos_emb = get_posemb(
          self, self.posemb, (h, w), c, "pos_embedding", jnp.float32
      )

  def embed_queries_and_hints(
      self,
      timesteps: int,
      query_points: chex.Array,  # Float["*B Q t 3"],
      query_padding: chex.Array,  # Float["*B Q t"]
  ) -> chex.Array:  # Float["*B T K c"]:
    """Computes embedded tokens for queries and video patches.

    First, it creates a tensor of queries embedding to perform imputation.
    This tensor has shape [B, T, Q, c]. Here is schematic view of how the
    tokens are allocated in the tensor:
    denote [M] - mask token, [U] - unknown token, and [XY] - a token with
    spatial pos embedding. The [U] token and [M] tokens are both learnable
    parameters of the model dimensionality. The [U] token is used to
    imitate the situation where the first query comes at an intermediate
    timestep, so the model is not tasked to predict anything at those positions.
    For example if the query point is at timestep 3, we will have 3 [U] tokens
    then 1 tokens with the coordinate infortaion written into the positional
    embedding, and the rest of [M] tokens. The model in turn will be tasked
    to only predict coordinates at timesteps where it was gives [M] tokens as
    input.

    Let's say, we have 5
    timesteps and 4 query tokens that have the following content:
    Q_0 = (t=0, x_0, y_0)
    Q_1 = (t=1, x_1, y_1)
    Q_2 = (t=0, x_2, y_2)
    Q_3 = (t=2, x_3, y_3)

    The resulting tensor will be filled like this: (hotizontal is the time axis
    and vertical is the track number axis)
    [XY]_0 [M]    [M]    [M] [M]
    [U]    [XY]_1 [M]    [M] [M]
    [XY]_2 [M]    [M]    [M] [M]
    [U]    [U]    [XY]_3 [M] [M]

    In the most general case, we allow the user to specify any number of hints
    per query. For example, say in the batch we have UP TO 3 hints per track
    (Q_jk stands for the k-th query of the j-th track):
    Q_00 = (t=0, x_00, y_00) Q_01 = (t=1, x_01, y_01)
    Q_10 = (t=1, x_10, y_10)
    Q_20 = (t=0, x_20, y_20) Q_21 = (t=1, x_21, y_21) Q_22 = (t=2, x_22, y_22)
    Q_30 = (t=2, x_30, y_30)

    The resulting tensor will be filled like this:

    [XY]_00 [XY]_01 [M]     [M] [M]
    [U]     [XY]_10 [M]     [M] [M]
    [XY]_20 [XY]_21 [XY]_22 [M] [M]
    [U]     [U]     [XY]_30 [M] [M]

    The idea is that at [M]-tokens, the model will be tasked to predict the
    corresponding (x, y)-coordinates and visibility flags and [U] tokens are
    treated as tokens where the information is not available so the model
    is not tasked to predict anything at those positions (to simulate the
    situation where the first query comes at an intermediate timestep).

    Args:
      timesteps: int: number of timesteps in the video
      query_points: [B Q t 3] point queries.
        we allow variable number of queries per track, therefore
        query_points[i, j] holds UP TO `t` valid queries for the j-th track.
      query_padding: [B Q t] this tensor indicates which queries are padded.
        query_padding[i, j, k] is 1 if the k-th query of the j-th
        track at batch element i is a valid query, and 0 otherwise.
    Returns:
      [B T Q c]
    """
    n, q, hints, _ = query_points.shape
    t = timesteps
    c = self.width
    match self.dtype_ssm:
      case "bfloat16":
        dtype = jnp.bfloat16
      case _:
        dtype = jnp.float32
    pixel_h, pixel_w = self.image_size
    tiled_point_query_tokens = jnp.tile(
        self.point_query_token, (n, q, hints, 1))
    tiled_mask_tokens = jnp.tile(self.mask_token, (n, t, q, 1))
    tiled_unknown_tokens = jnp.tile(self.unknown_token, (n, q, 1))
    posemb2d_full = get_posemb(
        self, self.posemb_full,
        (pixel_h * self.query_scale, pixel_w * self.query_scale), c,
        "pos_embedding_full", dtype
    )
    posemb2d_full_spatial = einops.rearrange(
        posemb2d_full, "1 (P E) c -> 1 P E c",
        P=pixel_h * self.query_scale,
        E=pixel_w * self.query_scale,
    )
    query_timesteps, query_positions = (
        query_points[..., :1], query_points[..., 1:])
    # mode="nearest" is the boundary strategy for interpolation.
    interp_fn = functools.partial(model_utils.interp, mode="nearest")
    # signature of interp_fn: [h w], [q 2] -> [q]
    # adds channels dimension: [h w c], [q 2] -> [q c]
    interp_fn = jax.vmap(interp_fn, in_axes=(-1, None), out_axes=-1)
    # adds batch dimension: [b h w c], [b q 2] -> [b q c]
    interp_fn = jax.vmap(interp_fn)
    # adds hints dimension: [b h w c], [b q t 2] -> [b q t c]
    interp_fn = jax.vmap(interp_fn, in_axes=(None, -2), out_axes=-2)
    # check_type(posemb2d_full_spatial, Float["1 P E c"])
    # check_type(query_positions, Float["*B Q t 2"])
    query_posemb_spatial = interp_fn(
        jnp.tile(posemb2d_full_spatial, (n, 1, 1, 1)),
        query_positions * self.query_scale)
    # check_type(query_posemb_spatial, Float["*B Q t c"])
    point_query_tokens = tiled_point_query_tokens + query_posemb_spatial
    # check_type(query_timesteps, Float["*B Q t 1"])
    query_timesteps = query_timesteps.astype(jnp.int32)
    query_padding = query_padding.astype(jnp.bool_)
    initial_point_query_tokens = point_query_tokens[..., 0, :]  # [B Q c]
    initial_query_timesteps = query_timesteps[..., 0, :]  # [B Q 1]
    # idea: implement multi-query prompting via unrolled loop over
    # the hints axis.
    # We initialize the query tokens tensor with the scaterring result
    # of the initial query tokens, which are guaranteed to be present.
    # then, at each timestep, we scatter every hint queries to the whole
    # query tokens tensor.

    # we add these conditions to make sure that the point queries are
    # within the range [0, timesteps]. This scatter the whole prefix
    # with a slice so the prefix_timestep can be equal to timesteps.
    prefix_timestep = initial_query_timesteps[..., 0]  # [B Q]
    prefix_timestep = jnp.clip(prefix_timestep, 0, timesteps)
    # First of all, we need to scatter query embeddings to the [B, T, Q, c]
    # tensor at their timesteps.
    temporal_query_tokens = index_utils.scatter_prefix(
        # pseudocode for this operation:
        # def scatter_prefix(target, mask, timestep, data):
        #   result = copy(target)
        #   for b in range(B):
        #     for q in range(Q):
        #       if mask[b, q]:
        #         result[b, :timestep, q] = data[b, q]
        #   return result
        # target, mask, timestep, data
        tiled_mask_tokens,  # [B T Q c] target
        query_padding[..., 0],  # [B Q] mask
        prefix_timestep,  # [B Q] timestep
        tiled_unknown_tokens,  # [B Q c] data
    )

    # here we update mask and clip the timestep.
    mask = query_padding[..., 0]  # [B Q]
    initial_query_timesteps = initial_query_timesteps[..., 0]  # [B Q]
    mask = jnp.logical_and(mask, initial_query_timesteps >= 0)  # [B Q]
    mask = jnp.logical_and(mask, initial_query_timesteps < timesteps)  # [B Q]
    # i.e. if initial query timstep is outside this range, we do not set
    # write positional embedding to the [B,T,Q,C] target tensor.

    initial_query_timesteps = jnp.clip(
        initial_query_timesteps, 0, timesteps - 1)
    # i.e. when we do write the positional embedding to the [B,T,Q,C] target
    # tensor, we write it to the correct timestep within the range [0, T-1].
    temporal_query_tokens = index_utils.scatter(
        # pseudocode for this operation:
        # def scatter(target, mask, timestep, data):
        #   result = copy(target)
        #   for b in range(B):
        #     for q in range(Q):
        #       if mask[b, q]:
        #         result[b, timestep, q] = data[b, q]
        #   return result
        temporal_query_tokens,  # [B T Q c] target
        mask,  # [B Q] mask
        initial_query_timesteps,  # [B Q] timestep
        initial_point_query_tokens,  # [B Q c] data
    )
    for hint_idx in range(1, hints):
      # temporal_query_tokens: [B T Q c] - target tensor
      current_mask = query_padding[..., hint_idx]  # [B Q] bool
      current_timesteps = query_timesteps[..., hint_idx, 0]  # [B Q]
      current_mask = jnp.logical_and(
          current_mask, current_timesteps >= 0)  # [B Q]
      current_mask = jnp.logical_and(
          current_mask, current_timesteps < timesteps
      )  # [B Q]
      current_timesteps = jnp.clip(current_timesteps, 0, timesteps - 1)
      current_tokens = point_query_tokens[..., hint_idx, :]  # [B Q c]
      temporal_query_tokens = index_utils.scatter(
          temporal_query_tokens, current_mask, current_timesteps, current_tokens
      )
    # check_type(temporal_query_tokens, Float["*B T Q c"])
    return temporal_query_tokens

  def __call__(
      self,
      video: chex.Array,  # Float["*B T H W 3"],
      query_points: chex.Array,  # Float["*B Q t 3"],
      query_padding: chex.Array,  # Float["*B Q t"],
      *,
      train=False,
  ):
    """Model forward pass used for training.

    This function does not take the SSM state as input and therefore does
    not support per-frame processing.

    Args:
      video: [B T H W 3]
      query_points: [B Q t 3]
      query_padding: [B Q t]
      train: bool

    Returns:
      x: intermediate tokens image tokens
      y: intermediate query tokens
    """
    out = {}
    # _, _, pixel_h, pixel_w, _ = video.shape
    video = jnp.asarray(video, self.dtype_mm)

    # Patch extraction
    x = out["stem"] = self.lin_proj(video)
    # check_type(x, Float["*B T h w c"])
    n, t, h, w, c = x.shape
    _, q, _, _ = query_points.shape
    temporal_query_tokens = self.embed_queries_and_hints(
        x.shape[1], query_points, query_padding,
    )
    if self.posemb == "learn":
      posemb2d = self.image_pos_emb
    else:
      posemb2d = get_posemb(
          self, self.posemb, (h, w), c, "pos_embedding", x.dtype
      )

    x = jnp.reshape(x, [n, t, h * w, c])
    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + posemb2d[:, None]
    x = jnp.concatenate([x, temporal_query_tokens], axis=2)
    # check_type(x, Float["*B T K c"])  # K = H*W + Q
    x = self.dropout_module(x, not train)
    x = jnp.reshape(x, [n, t, -1, c])

    x, out["encoder"] = self.encoder(x, deterministic=not train)
    x = jnp.reshape(x, [n, t, -1, c])

    if self.pool_type == "queries":
      x, y = (
          out["video_head_input"], out["query_head_input"]
      ) = (
          # in the embedding function we stack h*w image tokens and then
          # q query tokens. Therefore, we split them in the following way.
          # x.shape == [B T (h*w + q) c]
          x[:, :, :h*w, :], x[:, :, -q:, :])
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")
    x = einops.rearrange(x, "... (h w) c -> ... h w c", h=h, w=w)
    # out stores the intermediate tokens that we use
    # when we add loss in every layer.
    return x, y, out

  def forward_step(
      self,
      video: chex.Array,  # Float["*B T H W 3"],
      *,
      query_points: Optional[chex.Array] = None,  # B Q T 3
      query_padding: Optional[chex.Array] = None,  # B Q T
      state: Optional[TAPNextTrackingState] = None,
      train: bool = False,
  ) -> tuple[chex.Array, TAPNextTrackingState]:  # B T Q 3
    """Model forward pass used for per-frame processing."""

    if query_points is not None:
      x, _, out = self(video, query_points, query_padding)
      state = TAPNextTrackingState(
          hidden_state=out["encoder"]["ssm_block_cache"],
          step=x.shape[1],
          query_points=query_points,
          query_padding=query_padding
          if query_padding is not None
          else jnp.ones(query_points.shape[:-1], dtype=jnp.bool_),
      )
      return x, state
    elif state is None:
      raise ValueError("Cache and query points cannot both be None.")
    # therefore we assume here that state is not None and query_points is None.
    out = {}
    # _, _, pixel_h, pixel_w, _ = video.shape
    video = jnp.asarray(video, self.dtype_mm)

    # Patch extraction
    x = out["stem"] = self.lin_proj(video)
    # x --> B T h w c
    b, t, h, w, c = x.shape

    if state.hidden_state is None:
      raise ValueError("Hidden state is None.")
    n_tubes = state.hidden_state.rg_lru_state.shape[1]  # pylint: disable=attribute-error
    if n_tubes % b != 0:
      raise ValueError(
          f"Size of the cache {n_tubes} is not divisible by the batch size {b}"
      )
    hwq = n_tubes // b
    q = hwq - h * w
    if q <= 0:
      raise ValueError(
          f"Size of the cache patches {hwq} should be bigger"
          f" than the number of visual patches {h*w}"
      )
    query_points = state.query_points
    query_padding = state.query_padding
    # we "shift" query points time relatively to where we are in the video.
    # we allow them to become negative sometimes because this is handled in
    # embed_queries_and_hints
    query_points = jnp.concatenate([
        query_points[..., :1] - state.step,
        query_points[..., 1:],
    ], axis=-1)
    if len(query_points.shape) == 3:
      query_points = query_points[..., None, :]
    if query_padding is not None:
      if len(query_padding.shape) == 2:
        query_padding = query_padding[..., None]
    else:
      query_padding = jnp.ones(query_points.shape[:-1], dtype=jnp.bool_)
    temporal_query_tokens = self.embed_queries_and_hints(
        t, query_points, query_padding,
    )
    # temporal_query_tokens --> B T Q c
    if self.posemb == "learn":
      posemb2d = self.image_pos_emb
    else:
      posemb2d = get_posemb(
          self, self.posemb, (h, w), c, "pos_embedding", x.dtype
      )
    x = jnp.reshape(x, [b, t, h * w, c])
    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + posemb2d[:, None]
    x = jnp.concatenate([x, temporal_query_tokens], axis=2)
    # x --> B T K c, where K = H * W + Q
    x = self.dropout_module(x, not train)
    x = jnp.reshape(x, [b, t, -1, c])
    x, new_hidden_state = self.encoder(
        x, state.hidden_state, deterministic=not train)
    x = jnp.reshape(x, [b, t, -1, c])
    x = x[:, :, -q:, :]
    # x --> B T Q c
    new_hidden_state = new_hidden_state["ssm_block_cache"]
    new_state = TAPNextTrackingState(
        hidden_state=new_hidden_state,
        step=state.step + t,
        query_points=state.query_points,
        query_padding=state.query_padding,
    )
    return x, new_state


def Model(*, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return MaskedSequenceDecoder(**{**decode_variant(variant), **kw})


def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      "width": {
          "mu": 32,
          "Ti": 192,
          "S": 384,
          "M": 512,
          "B": 768,
          "L": 1024,
          "So400m": 1152,
          "H": 1280,
          "g": 1408,
          "g-opt": 1536,
          "G": 1664,
          "G-opt": 1536,
          "e": 1792,
      }[v],
      "depth": {
          "mu": 1,
          "Ti": 12,
          "S": 12,
          "M": 12,
          "B": 12,
          "L": 24,
          "So400m": 27,
          "H": 32,
          "g": 40,
          "g-opt": 40,
          "G": 48,
          "G-opt": 48,
          "e": 56,
      }[v],
      "mlp_dim": {
          "mu": 128,
          "Ti": 768,
          "S": 1536,
          "M": 2048,
          "B": 3072,
          "L": 4096,
          "So400m": 4304,
          "H": 5120,
          "g": 6144,
          "g-opt": 6144,
          "G": 8192,
          "G-opt": 8192,
          "e": 15360,
      }[v],
      "num_heads": {
          "mu": 2,
          "Ti": 3,
          "S": 6,
          "M": 8,
          "B": 12,
          "L": 16,
          "So400m": 16,
          "H": 16,
          "g": 16,
          "g-opt": 16,
          "G": 16,
          "G-opt": 16,
          "e": 16,
      }[v],
      **patch,
  }

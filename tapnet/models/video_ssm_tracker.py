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

"""TAPNext model."""

from __future__ import annotations

from typing import Any, Optional

import chex
import einops
import flax
from flax import linen as nn
import jax.nn as jnn
import jax.numpy as jnp

from tapnet.models import ssm_vit


@flax.struct.dataclass
class TrackerResults:
  """Container for TAPNext results."""

  # Estimated point tracks trajectories over the video.
  tracks: chex.Array  # Float["*B Q T 2"]
  track_logits: chex.Array  # Float["*B Q T 512"]
  # Raw visibility predictions in (-inf, inf), i.e. pre-sigmoid.
  visible_logits: chex.Array  # Float["*B Q T 1"]
  # Estimated point tracks trajectories, intermediately.
  intermediate_tracks: chex.Array  # Sequence[Float["*B Q T 2"]]
  intermediate_track_logits: chex.Array  # Sequence[Float["*B Q T 512"]]
  # Raw visibility predictions in (-inf, inf), intermediately.
  intermediate_visible_logits: chex.Array  # Sequence[Float["*B Q T 1"]]
  state: Optional[Any] = None

  # Hard predictions in {0, 1}.
  @property
  def visible(self) -> Any:  #  Float["*B Q T 1"]
    return (self.visible_logits > 0).astype(jnp.float32)


class TAPNextTracker(nn.Module):
  """VideoSSM Tracker model."""

  backbone: nn.Module

  soft_argmax_threshold: int = 20
  softmax_temperature: float = 0.5
  # "argmax", "softargmax", or "trunc_softargmax"
  head_mode: str = "mlp"  # "linear" or "mlp"

  bilinear_interp_with_depthwise_conv: bool = False
  model_key: str = "+mlp"

  def setup(self):
    if self.head_mode == "linear":
      self.visible_head = nn.Dense(features=1)
      if self.coord_prediction_type == "scalar":
        self.coordinate_head = nn.Dense(features=2)
      elif self.coord_prediction_type == "quantized":
        self.coordinate_head = nn.Dense(features=512)
      else:
        raise ValueError(
            f"Unknown coord_prediction_type: '{self.coord_prediction_type=}'")
    elif self.head_mode == "mlp":
      make_mlp = lambda out, inner: nn.Sequential([
          nn.Dense(inner),
          nn.LayerNorm(),
          nn.gelu,
          nn.Dense(inner),
          nn.LayerNorm(),
          nn.gelu,
          nn.Dense(out),
      ])
      self.visible_head = make_mlp(1, 256)
      self.coordinate_head = make_mlp(512, 256)

  def prediction_heads(
      self, query_points_features: chex.Array  # Float["*B T Q C"]
  ) -> tuple[chex.Array, chex.Array, chex.Array]:
    #  Float["*B T Q 2"], Float["*B T Q 512"],
    #  Float["*B T Q 1"]
    # Cast the features to float32 just before readout heads are used, in
    # case we are suing bfloat16.
    query_points_features = jnp.astype(query_points_features, jnp.float32)
    position = self.coordinate_head(query_points_features)
    position_x, position_y = jnp.split(position, 2, axis=-1)
    argmax_x = jnp.argmax(position_x, axis=-1, keepdims=True)
    argmax_y = jnp.argmax(position_y, axis=-1, keepdims=True)
    index_x = jnp.arange(position_x.shape[-1])[None, None, None, :]
    index_x = jnp.tile(index_x, (*argmax_x.shape[:-1], 1))
    index_y = jnp.arange(position_y.shape[-1])[None, None, None, :]
    index_y = jnp.tile(index_y, (*argmax_y.shape[:-1], 1))
    mask_x = jnp.abs(argmax_x - index_x) <= self.soft_argmax_threshold
    mask_y = jnp.abs(argmax_y - index_y) <= self.soft_argmax_threshold
    mask_x = mask_x.astype(jnp.float32)
    mask_y = mask_y.astype(jnp.float32)
    probs_x = jnn.softmax(position_x * self.softmax_temperature, axis=-1)
    probs_y = jnn.softmax(position_y * self.softmax_temperature, axis=-1)
    probs_x = probs_x * mask_x
    probs_y = probs_y * mask_y
    probs_x = probs_x / jnp.sum(probs_x, axis=-1, keepdims=True)
    probs_y = probs_y / jnp.sum(probs_y, axis=-1, keepdims=True)
    tracks_x = jnp.sum(probs_x * index_x, axis=-1)[..., None]
    tracks_y = jnp.sum(probs_y * index_y, axis=-1)[..., None]
    tracks = jnp.concatenate([tracks_x, tracks_y], axis=-1)
    tracks += 0.5
    track_logits = position
    visible_logits = self.visible_head(query_points_features)

    return tracks, track_logits, visible_logits

  def __call__(
      self,
      video: chex.Array,  # Float["*B T H W 3"],
      query_points: chex.Array,  # Float["*B Q t 3"] | Float["*B Q 3"],
      query_padding: Optional[
          chex.Array
      ] = None,  # [Bool["*B Q t"] | Bool["*B Q"]] = None,
      return_cache: bool = False,
  ) -> TrackerResults:
    """This is the forward function used for training on videos.

    It takes the video and a set of query points as args. It then runs the
    backbone on the video and query points and returns the results from all
    intermediate and output layers for per-layer loss computation.

    The forward_step function (defined below) is similar to call, but also
    optionally takes the state as an arg enabling us to run the model online
    in a per-frame manner.

    Queries are in t, x, y format. t is an integer corresponding to the
    frame number. x and y are the unnormalized, sub-pixelcoordinates of the
    query in the image plane. Please see embed_queries_and_hints in ssm_vit.py
    for more details.


    Args:
      video: [B T H W 3]
      query_points: [B Q t 3] or [B Q 3]
      query_padding: [B Q t] or [B Q]
      return_cache: bool

    Returns:
      TrackerResults
    """
    batch_size, seq_len, _, _, _ = video.shape
    # query_points is [B Q t 3] or [B Q 3]
    # when it is [B Q 3], we assume we have one point for each track
    # when it is [B Q t 3], we assume we have UP TO t points for each track
    # (we call them hints / or multi-query prompting)
    if query_padding is None:
      query_padding = jnp.ones(query_points.shape[:-1], dtype=jnp.bool_)
    if len(query_points.shape) == 3:
      query_points = query_points[..., None, :]
    if len(query_padding.shape) == 2:
      query_padding = query_padding[..., None]
    query_padding = query_padding.astype(jnp.float32)
    if self.backbone.dtype_ssm == "bfloat16":
      video = video.astype(jnp.bfloat16)
      query_points = query_points.astype(jnp.bfloat16)
      query_padding = query_padding.astype(jnp.bfloat16)

    video_features, query_points_features, outputs = self.backbone(
        video, query_points, query_padding
    )
    # The tokens representing image patches are not used in the tracking loss.
    del video_features

    query_tokens = query_points_features.shape[2]
    intermediate_tracks = []
    intermediate_track_logits = []
    intermediate_visible_logits = []
    for lyr in range(self.backbone.depth):
      intermediate_feature = (
          outputs["encoder"][f"block{lyr:02d}"][
              "vit_block_intermediates"][self.model_key]
      )
      intermediate_query_points_features = (
          intermediate_feature[:, -query_tokens:]
      )
      intermediate_query_points_features = einops.rearrange(
          intermediate_query_points_features,
          "... (B T) N d -> ... B T N d",
          B=batch_size,
          T=seq_len,
      )
      # check_type(intermediate_query_points_features, Float["*B T Q C"])
      tracks, track_logits, visible_logits = (
          self.prediction_heads(intermediate_query_points_features)
      )
      tracks = einops.rearrange(tracks, "... T Q d -> ... Q T d")
      track_logits = einops.rearrange(
          track_logits, "... T Q v -> ... Q T v")
      visible_logits = einops.rearrange(
          visible_logits, "... T Q k -> ... Q T k")
      intermediate_tracks.append(tracks)
      intermediate_track_logits.append(track_logits)
      intermediate_visible_logits.append(visible_logits)
    # check_type(video_features, Float["*B T h w C"])
    # check_type(query_points_features, Float["*B T Q C"])

    tracks, track_logits, visible_logits = (
        self.prediction_heads(query_points_features)
    )
    tracks = einops.rearrange(tracks, "... T Q d -> ... Q T d")
    track_logits = einops.rearrange(
        track_logits, "... T Q v -> ... Q T v")
    visible_logits = einops.rearrange(
        visible_logits, "... T Q k -> ... Q T k")
    return TrackerResults(
        intermediate_tracks=intermediate_tracks,
        intermediate_track_logits=intermediate_track_logits,
        intermediate_visible_logits=intermediate_visible_logits,
        tracks=tracks,
        track_logits=track_logits,
        visible_logits=visible_logits,
        state=outputs["encoder"]["ssm_block_cache"] if return_cache else None,
    )

  def forward_step(
      self,
      frames: chex.Array,  # Float["*B T H W 3"],
      *,
      query_points: Optional[chex.Array] = None,  # Float["*B Q t 3"]] = None,
      query_padding: Optional[chex.Array] = None,  # Float["*B Q t"]] = None,
      state: Optional[Any] = None,
  ) -> TrackerResults:
    if state is None and query_points is None:
      raise ValueError("Cache and query points cannot both be None.")
    if query_points is not None:
      track_results = self(
          frames, query_points, query_padding, return_cache=True
      )
      return TrackerResults(
          intermediate_tracks=track_results.intermediate_tracks,
          intermediate_track_logits=track_results.intermediate_track_logits,
          intermediate_visible_logits=track_results.intermediate_visible_logits,
          tracks=track_results.tracks,
          track_logits=track_results.track_logits,
          visible_logits=track_results.visible_logits,
          state=ssm_vit.TAPNextTrackingState(
              hidden_state=track_results.state,
              step=frames.shape[1],
              query_points=query_points,
              query_padding=query_padding,
          ),
      )
    x, new_state = self.backbone.forward_step(frames, state=state)
    # check_type(x, Float["*B T Q C"])
    tracks, track_logits, visible_logits = (
        self.prediction_heads(x)
    )
    tracks = einops.rearrange(tracks, "... T Q d -> ... Q T d")
    track_logits = einops.rearrange(
        track_logits, "... T Q v -> ... Q T v")
    visible_logits = einops.rearrange(
        visible_logits, "... T Q k -> ... Q T k")
    return TrackerResults(
        intermediate_tracks=[],
        intermediate_track_logits=[],
        intermediate_visible_logits=[],
        tracks=tracks,
        track_logits=track_logits,
        visible_logits=visible_logits,
        state=new_state,
    )

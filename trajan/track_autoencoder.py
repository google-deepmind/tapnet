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

"""Track autoencoder modules for TRAJAN."""

from __future__ import annotations

import functools
from typing import Any, NotRequired, TypedDict

import einops
import flax
from flax import linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp

from tapnet.trajan import attention


class SinusoidalEmbedding(nn.Module):
  """Create Sinusoidal embeddings for scalars."""

  num_frequencies: int

  @nn.compact
  def __call__(
      self,
      inputs,  # float["*B coords"],
  ):  # -> float["*B coords*2*{self.num_frequencies}"]
    scales = jnp.asarray([2 ** (i / 3) for i in range(self.num_frequencies)])
    # (... coords) @ (num_frequencies) -> (... coords num_frequencies)
    x = jnp.einsum("...,b->...b", inputs, scales)
    # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
    # (vectorize `cos` into a single `sin` call)
    # Instead of interleave sin and cos
    # (`sin(x0), cos(x0), sin(x1), cos(x1),...`), they are concatenated
    # (`sin(x0), sin(x1), ..., cos(x0), cos(x1), ...`)
    outputs = jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))
    outputs = einops.rearrange(outputs, "... coords d -> ... (coords d)")
    return outputs


class ParamStateInit(nn.Module):
  """Fixed, learnable state initialization."""

  shape: tuple[int, ...]

  @nn.compact
  def __call__(
      self,
      batch_shape,
  ):  # -> float["*B n d"]
    init_fn = functools.partial(nn.initializers.normal, stddev=1.0)
    param = self.param("state_init", init_fn(), self.shape)
    return jnp.broadcast_to(array=param, shape=batch_shape + param.shape)


class TrackAutoEncoderInputs(TypedDict):
  """Track autoencoder inputs.

  Attributes:
    query_points: The (t, x, y) locations of a set of query points on initial
      frame. The decoder predicts the location and visibility of these query
      points for T frames into the future.
    boundary_frame: Int specifying the first frame of any padding in the support
      tracks.  Track values starting on this frame will be masked out of the
      attention for transformers.
  """

  query_points: NotRequired[Any]  # NotRequired[float["*B Q 3"]]
  boundary_frame: Any  # int["*B"]


@flax.struct.dataclass
class TrackAutoEncoderResults:
  """Container for track autoencoder results.

  Attributes:
    tracks: The (x, y) locations predicted locations.
    visible_logits: The raw visibility prediction logits.
    certain_logits: The raw certainty predictions logits.
    visible: Predicted visibility probability computed from visible logits.
    certain: Predicted certainty probability computed from the logits.
    visible_and_certain: Predicted visibility and certainty combined.
  """

  # Estimated point tracks over the video.
  tracks: Any  # float["*B Q T 2"]
  # Raw visibility predictions in (-inf, inf), i.e. pre-sigmoid.
  visible_logits: Any  # float["*B Q T 1"]
  # Raw certainty predictions in (-inf, inf), i.e. pre-sigmoid.
  certain_logits: Any  # float["*B Q T 1"]

  # Hard predictions in {0, 1}.
  @property
  def visible(self):  # -> float["*B Q T 1"]
    return (self.visible_logits > 0).astype(jnp.float32)

  @property
  def certain(self):  # -> float["*B Q T 1"]
    return (self.certain_logits > 0).astype(jnp.float32)

  @property
  def visible_and_certain(self):  # -> float["*B Q T 1"]
    visible = jnn.sigmoid(self.visible_logits)
    certain = jnn.sigmoid(self.certain_logits)
    return ((visible * certain) > 0.5).astype(jnp.float32)


@flax.struct.dataclass
class TrackAutoEncoderDecoderContext:
  """Container for track autoencoder decoder context."""

  decoder_query: Any  # float["*B Q FF"]
  query_frame: Any  # int["*B QS"]
  boundary_frame: Any  # int["*B"] | None


class TrackAutoEncoder(nn.Module):
  """A track autoencoder based on transformers."""

  num_output_frames: int = 150

  num_latent_tokens: int = 128
  latent_token_dim: int = 64

  # Number of frequencies for sinusoidal embedding
  num_frequencies: int = 32
  # Rescale tracks before applying sinusoidal embedding
  track_scale_factor: float = 1.0
  time_scale_factor: float = 150.0
  # Shared token dimension for image and track tokens
  track_token_dim: int = 256
  encoder_latent_dim: int = 512
  decoder_num_channels: int = 1024

  decoder_scan_chunk_size: int | None = None

  def setup(self):
    self.initializer = ParamStateInit(
        shape=(self.num_latent_tokens, self.encoder_latent_dim),
    )
    self.track_token_projection = nn.Dense(self.track_token_dim)
    self.sinusoidal_embedding = SinusoidalEmbedding(
        num_frequencies=self.num_frequencies
    )
    self.compressor = nn.Dense(self.latent_token_dim)
    self.decompressor = nn.Dense(self.decoder_num_channels - 128)
    self.input_readout_token = ParamStateInit(shape=(1, self.track_token_dim))
    self.input_track_transformer = attention.ImprovedTransformer(
        qkv_size=64 * 8,
        num_heads=8,
        mlp_size=1024,
        num_layers=2,
    )
    self.tracks_to_latents = attention.ImprovedTransformer(
        qkv_size=64 * 8,
        num_heads=8,
        mlp_size=2048,
        num_layers=6,
    )
    self.decompress_attn = attention.ImprovedTransformer(
        qkv_size=64 * 8,
        num_heads=8,
        mlp_size=2048,
        num_layers=3,
    )
    self.track_readout_attn = attention.ImprovedTransformer(
        qkv_size=64 * 8,
        num_heads=8,
        mlp_size=1024,
        num_layers=4,
    )
    self.query_encoder = nn.Dense(self.decoder_num_channels)
    self.track_predictor = nn.Dense(self.num_output_frames * 4)

  def encode_point_identities(
      self, query_points  # float["*B Q 2"]
  ):  # -> float["*B Q {4*self.num_frequencies}"]
    """Encode point identities as embeddings of corresponding query point."""
    # Note: we only take 2 dimensions because we don't currently encode time.
    # Get track identities embeddings
    queries = query_points / self.track_scale_factor
    track_identities = self.sinusoidal_embedding(queries)
    return track_identities

  def embed_track_pos_visible(
      self,
      tracks,  # float["*B Q T 2"]
      visible,  # float["*B Q T 1"]
  ):  # -> float["*B Q T {6*self.num_frequencies}"]
    # Embed point coordinates
    fr_id = jnp.arange(tracks.shape[-2]) / tracks.shape[-2]
    fr_id = jnp.broadcast_to(
        fr_id[jnp.newaxis, jnp.newaxis, :, jnp.newaxis], visible.shape
    )
    tracks = jnp.concatenate([tracks, fr_id], axis=-1)
    point_coords_embedding = self.sinusoidal_embedding(
        tracks / self.track_scale_factor
    )

    # Concatenate position and visibility
    track_embeddings = point_coords_embedding

    return track_embeddings

  def encode_tracks(
      self,
      tracks,  # float["*B Q T 2"]
      visible,  # float["*B Q T 1"]
      restart,  # int["*B"] | None
  ):  # -> float["*B Q C"]
    track_embeddings = self.embed_track_pos_visible(
        tracks=tracks, visible=visible
    )
    track_tokens = track_embeddings
    track_tokens = self.track_token_projection(track_tokens)

    time = jnp.arange(visible.shape[2])
    # partition gets broadcast to ["*B 1 1 T+1"]
    partition = time < restart[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    visible = jnp.array(visible[..., 0], jnp.bool)

    # visibility_mask is [*B QS T+1 T+1]
    visibility_mask = (
        jnp.ones_like(visible[..., jnp.newaxis]) * visible[..., jnp.newaxis, :]
    )

    track_tokens = self.input_track_transformer(
        track_tokens, qq_mask=partition * visibility_mask
    )
    return jnp.sum(
        track_tokens * visible[..., jnp.newaxis], axis=-2
    ) / jnp.maximum(1.0, jnp.sum(visible[..., jnp.newaxis], axis=-2))

  def encode(self, inputs):  # -> float["*B N CL"]
    # Get video tokens
    support_track_tokens = self.encode_tracks(
        tracks=inputs["support_tracks"],
        visible=inputs["support_tracks_visible"],
        restart=inputs["boundary_frame"],
    )

    latents = self.initializer(batch_shape=(inputs["support_tracks"].shape[0],))
    latents = self.tracks_to_latents(latents, support_track_tokens)

    latents = self.compressor(latents)
    return latents

  @nn.remat
  def get_decoder_context(self, inputs):
    # Get decoder query with query point identities
    if "query_points" in inputs:
      decoder_query = inputs["query_points"][..., 1:]
      query_frame = jnp.array(
          jnp.round(inputs["query_points"][..., 0]), jnp.int32
      )
    else:
      # default to a grid
      grid_centers = jnp.arange(32) / 32.0 + 1.0 / 64.0
      query_x, query_y = jnp.meshgrid(grid_centers, grid_centers)
      decoder_query = jnp.reshape(
          jnp.stack([query_x, query_y], axis=-1), [-1, 2]
      )
      decoder_query = jnp.broadcast_to(
          decoder_query,
          inputs["support_tracks"].shape[:-3] + decoder_query.shape,
      )
      query_frame = jnp.array(decoder_query[..., 0], jnp.int32) * 0
    decoder_query = self.encode_point_identities(query_points=decoder_query)
    return TrackAutoEncoderDecoderContext(
        decoder_query=decoder_query,
        query_frame=query_frame,
        boundary_frame=inputs["boundary_frame"],
    )

  def append_time_feat(
      self,
      latents,  # float["*B Q N C"]
      query_frame,  # int["*B Q"]
  ):  # -> float["*B Q N CC"]:
    assert latents.shape[-1] == (1024 - 128)

    def get_eye(idx):
      return jnp.eye(128, latents.shape[-1], idx * 5)

    for _ in range(query_frame.ndim):
      get_eye = jax.vmap(get_eye)
    multiplier = get_eye(query_frame)
    to_append = jnp.einsum("... N C , ... D C -> ... N D", latents, multiplier)
    return jnp.concatenate([latents, to_append], axis=-1)

  @nn.remat
  def decode(
      self,
      latents,  # float["*B N CL"],
      decoder_context: TrackAutoEncoderDecoderContext,
      discretize: bool = True,
  ) -> TrackAutoEncoderResults:
    latents = jnp.clip(latents, -1.0, 1.0)
    if discretize:
      latents_disc = jnp.round(latents * 128.0) / 128.0
      rng = jax.random.PRNGKey(0)
      latents_disc = (
          latents_disc
          + jax.random.uniform(rng, latents_disc.shape) / 128.0
          - 1.0 / 256.0
      )
      latents = latents - jax.lax.stop_gradient(latents - latents_disc)
    latents = self.decompressor(latents)
    latents = self.decompress_attn(latents)

    queries = jnp.concatenate(
        [
            decoder_context.decoder_query,
            decoder_context.query_frame[..., jnp.newaxis]
            // self.time_scale_factor,
        ],
        axis=-1,
    )
    point_coords_embedding = self.query_encoder(
        self.sinusoidal_embedding(queries / self.track_scale_factor)
    )
    latents = jnp.tile(
        latents[..., jnp.newaxis, :, :],
        (1,) * len((latents.shape[0],))
        + (point_coords_embedding.shape[-2], 1, 1),
    )
    latents = self.append_time_feat(latents, decoder_context.query_frame)
    latents = jnp.concatenate(
        [point_coords_embedding[..., jnp.newaxis, :], latents], axis=2
    )
    out = self.track_readout_attn(latents)
    out = out[..., 0, :]
    out = self.track_predictor(out)
    num_frames = self.num_output_frames
    tracks = jnp.stack(
        [out[..., :num_frames], out[..., num_frames : 2 * num_frames]], axis=-1
    )
    visible_logits = out[..., 2 * num_frames : 3 * num_frames, jnp.newaxis]
    certain_logits = out[..., 3 * num_frames :, jnp.newaxis]

    return TrackAutoEncoderResults(
        tracks=tracks,
        visible_logits=visible_logits,
        certain_logits=certain_logits,
    )

  def __call__(self, inputs):
    """Forward pass for autoencoder."""
    latents = self.encode(inputs)
    if self.decoder_scan_chunk_size is None:
      decoder_context = self.get_decoder_context(inputs)
      outputs = self.decode(latents=latents, decoder_context=decoder_context)
    else:

      def scan_fn(tr_enc, carry, qp):
        autoencoder_inputs = TrackAutoEncoderInputs(
            query_points=qp + carry, boundary_frame=inputs["boundary_frame"]
        )
        decoder_context = tr_enc.get_decoder_context(autoencoder_inputs)
        res = tr_enc.decode(latents, decoder_context)
        carry = jnp.sum(res.tracks) > 1e20
        return carry, res

      scan_fn2 = nn.scan(
          scan_fn,
          variable_broadcast="params",
          split_rngs={"params": False, "default": True},
          in_axes=-3,
          out_axes=-4,
      )
      h = self.decoder_scan_chunk_size
      _, preds = scan_fn2(
          self,
          False,
          einops.rearrange(
              inputs["query_points"], "... (Q H) C -> ... Q H C", H=h
          ),
      )
      outputs = jax.tree_util.tree_map(
          lambda x: einops.rearrange(x, "... Q H T C -> ... (Q H) T C", H=h),
          preds,
      )

    outputs = TrackAutoEncoderResults(
        tracks=outputs.tracks,
        visible_logits=outputs.visible_logits,
        certain_logits=outputs.certain_logits,
    )

    return outputs

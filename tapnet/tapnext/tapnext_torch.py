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

"""TAPNext implementation in torch."""

import dataclasses
from typing import List

import einops
import numpy as np
from tapnet.tapnext import tapnext_lru_modules
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vision_transformer


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=np.float32):
  """Follows the MoCo v3 logic."""
  y, x = np.mgrid[:h, :w]

  assert width % 4 == 0, 'Width must be mult of 4 for sincos posemb'
  omega = np.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = np.einsum('m,d->md', y.flatten(), omega)
  x = np.einsum('m,d->md', x.flatten(), omega)
  pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
  return np.asarray(pe, dtype)[None, :, :]


class TRecViTBlock(nn.Module):
  """A block proposed by https://arxiv.org/abs/2412.14294."""

  def __init__(self, depth, width, num_heads, lru_width, dtype, device):
    super().__init__()
    self.ssm_block = tapnext_lru_modules.ResidualBlock(
        width=width,
        mlp_expanded_width=width * 4,
        num_heads=num_heads,
        lru_width=lru_width,
        final_w_init_variance_scale=2.0 / depth,
        dtype=dtype,
        device=device,
    )
    self.vit_block = vision_transformer.EncoderBlock(
        num_heads=num_heads,
        mlp_dim=width * 4,
        hidden_dim=width,
        attention_dropout=0.0,
        dropout=0.0,
    )

  def forward(self, x, cache=None, use_linear_scan=True):
    b, t, n, _ = x.shape
    x = einops.rearrange(x, 'b t n c -> (b n) t c')
    x, ssm_cache = self.ssm_block(x, cache, use_linear_scan=use_linear_scan)
    x = einops.rearrange(x, '(b n) t c -> (b t) n c', b=b, n=n)
    x = self.vit_block(x)
    x = einops.rearrange(x, '(b t) n c -> b t n c', b=b, t=t)
    return x, ssm_cache


@dataclasses.dataclass
class TAPNextTrackingState:
  """State for TAPNext."""

  step: int
  query_points: torch.Tensor  # Float["*B Q 3"]
  hidden_state: List[tapnext_lru_modules.RecurrentBlockCache] = None


class TAPNext(nn.Module):
  """TAPNext implementation in pytorch."""

  def __init__(
      self,
      image_size,
      width=768,
      patch_size=(8, 8),
      num_heads=12,
      lru_width=768,
      depth=12,
      use_checkpointing=False,
  ):
    super().__init__()
    self.width = width
    self.patch_size = patch_size
    self.use_checkpointing = use_checkpointing
    self.image_size = image_size

    self.lin_proj = nn.Conv2d(
        in_channels=3,
        out_channels=self.width,
        kernel_size=self.patch_size,
        stride=self.patch_size,
    )
    self.blocks = nn.ModuleList([
        TRecViTBlock(
            depth=depth,
            width=width,
            num_heads=num_heads,
            lru_width=lru_width,
            dtype=torch.float32,
            device='cuda',
        )
        for _ in range(depth)
    ])
    self.encoder_norm = nn.LayerNorm(self.width)
    self.mask_token = nn.Parameter(
        torch.zeros((1, 1, 1, self.width)), requires_grad=True
    )
    self.unknown_token = nn.Parameter(
        torch.zeros((1, 1, self.width)), requires_grad=True
    )
    self.point_query_token = nn.Parameter(
        torch.zeros((1, 1, 1, self.width)), requires_grad=True
    )
    h = self.image_size[0] // self.patch_size[0]
    w = self.image_size[1] // self.patch_size[1]
    c = self.width
    self.image_pos_emb = nn.Parameter(
        torch.zeros((1, h * w, c)), requires_grad=True
    )
    self.register_buffer(
        'query_pos_embed',
        torch.tensor(
            posemb_sincos_2d(self.image_size[0], self.image_size[1], c)
        ),
    )
    self.visible_head = nn.Sequential(
        nn.Linear(width, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 1),
    )
    self.coordinate_head = nn.Sequential(
        nn.Linear(width, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 512),
    )

  def embed_queries(self, timesteps, query_points):
    b, q, _ = query_points.shape
    t = timesteps
    c = self.width
    tiled_point_query_tokens = self.point_query_token.repeat(b, 1, q, 1)
    mask_tokens = self.mask_token.repeat(b, t, q, 1)
    unknown_tokens = self.unknown_token.repeat(b, t, q, 1)
    query_pos_embed = self.query_pos_embed.view(
        1, self.image_size[0], self.image_size[1], c
    ).repeat(b, 1, 1, 1)
    #  [B Q t 3]
    query_timesteps, query_positions = (
        query_points[..., :1],
        query_points[..., 1:],
    )
    # grid sample expects coordinates in [-1, 1]
    query_pos_embed_spatial = F.grid_sample(
        query_pos_embed.permute(
            0, 3, 2, 1
        ),  # we swap h and w due to how grid_sample works
        (
            query_positions.unsqueeze(1)
            / torch.tensor(self.image_size, device=query_positions.device)
        )
        * 2
        - 1,
        align_corners=False,
    ).permute(
        0, 2, 3, 1
    )  # [b Q t c]
    point_query_tokens = tiled_point_query_tokens + query_pos_embed_spatial
    # NOTE: query hints are only implemented in jax but not in pytorch
    # the two masks below are used to not add point query (if queried later in
    # online tracking)
    if t == 1:  # online tracking
      mask_and_query_tokens = torch.where(
          query_timesteps.unsqueeze(1) == 0, point_query_tokens, mask_tokens
      )
    else:
      queries_are_late = query_timesteps >= t
      queries_are_early = query_timesteps < 0
      mask_and_query_tokens = mask_tokens.scatter(
          dim=1,
          index=query_timesteps.unsqueeze(1)
          .long()
          .clamp(0, t - 1)
          .repeat(1, 1, 1, c),
          src=point_query_tokens,
      )
      mask_and_query_tokens = torch.where(
          (queries_are_late | queries_are_early).unsqueeze(1),
          mask_tokens,
          mask_and_query_tokens,
      )
    is_unknown_token = torch.arange(t, device=query_points.device)[
        None, :, None, None
    ] < query_timesteps.unsqueeze(1)
    mask_and_query_tokens = torch.where(
        is_unknown_token, unknown_tokens, mask_and_query_tokens
    )
    return mask_and_query_tokens

  def prediction_heads(self, x):
    soft_argmax_threshold = 20
    softmax_temperature = 0.5
    track_logits = self.coordinate_head(x.float())
    position_x, position_y = track_logits.chunk(2, dim=-1)
    argmax_x, argmax_y = position_x.argmax(
        dim=-1, keepdim=True
    ), position_y.argmax(dim=-1, keepdim=True)
    index = torch.arange(position_x.shape[-1], device=x.device).repeat(
        *argmax_x.shape[:-1], 1
    )
    mask_x = (torch.abs(argmax_x - index) <= soft_argmax_threshold).float()
    mask_y = (torch.abs(argmax_y - index) <= soft_argmax_threshold).float()
    probs_x = F.softmax(position_x * softmax_temperature, dim=-1) * mask_x
    probs_y = F.softmax(position_y * softmax_temperature, dim=-1) * mask_y
    probs_x = probs_x / probs_x.sum(dim=-1, keepdim=True)
    probs_y = probs_y / probs_y.sum(dim=-1, keepdim=True)
    tracks_x = torch.sum(probs_x * index, dim=-1)[..., None]
    tracks_y = torch.sum(probs_y * index, dim=-1)[..., None]
    tracks = torch.cat([tracks_x, tracks_y], axis=-1)
    tracks += 0.5
    visible_logits = self.visible_head(x)
    return tracks, track_logits, visible_logits

  def forward(self, video, query_points=None, state=None):
    # video.shape
    b, t, _, _, _ = video.shape
    # [b, t, h, w, 3] -> [b, t, 3, h, w]
    video_tokens = self.lin_proj(
        einops.rearrange(video, 'b t h w c -> (b t) c h w')
    )
    _, _, h, w = video_tokens.shape
    video_tokens = einops.rearrange(
        video_tokens, '(b t) c h w -> b t (h w) c', b=b, t=t
    )
    video_tokens = video_tokens + self.image_pos_emb.unsqueeze(0)
    if state is not None:
      # in online tracking, we put query "back in time"
      if query_points is None:
        query_points = state.query_points
      query_points = torch.cat(
          [query_points[..., :1] - state.step, query_points[..., 1:]], dim=-1
      )
      step = state.step
    else:
      step = 0
    point_tokens = self.embed_queries(t, query_points)  # [b t Q c]
    x = torch.cat([video_tokens, point_tokens], dim=2)  # [b t (h * w + Q) c]
    ssm_cache = []
    use_linear_scan = not self.training
    for blk, cache in zip(
        self.blocks, state.hidden_state if state is not None else [None] * 12
    ):
      if self.use_checkpointing:
        x, ssm_cache_layer = torch.utils.checkpoint.checkpoint(
            blk, x, cache, use_linear_scan, use_reentrant=False
        )
      else:
        x, ssm_cache_layer = blk(
            x, cache=cache, use_linear_scan=use_linear_scan
        )
      ssm_cache.append(ssm_cache_layer)
    x = self.encoder_norm(x)
    video_tokens, point_tokens = x[:, :, : h * w, :], x[:, :, h * w :, :]
    return (
        *self.prediction_heads(point_tokens),
        TAPNextTrackingState(
            step=step + t,
            query_points=state.query_points
            if state is not None
            else query_points,
            hidden_state=ssm_cache,
        ),
    )

torch.export.register_dataclass(TAPNextTrackingState)


def flatten_tracking_state(state, _):
  return (
      state.step,
      state.query_points,
      [[st for st in h] for h in state.hidden_state],
  )


torch.fx._pytree.register_pytree_flatten_spec(  # pylint: disable=protected-access
    TAPNextTrackingState, flatten_tracking_state
)

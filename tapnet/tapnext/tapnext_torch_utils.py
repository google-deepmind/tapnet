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

"""Utils for TAPNext torch implementation."""

import re
import numpy as np
import torch
import torch.nn.functional as F


def get_window(coord, softmax, radius: int = 8):
  b = coord.shape[0]
  start = torch.floor(coord - radius - 0.5).int()
  start.clamp_(min=0)
  indices = start + torch.arange(radius * 2 + 1, device=softmax.device).repeat(
      b, 1
  )
  # this is to simulate one corner case of jax implementation
  shift = (indices.max(1).values - softmax.shape[1] + 1).clamp(min=0)
  indices -= shift.unsqueeze(1)
  softmax = softmax.gather(dim=1, index=indices)
  return softmax, indices + 0.5


def tracker_certainty(coord_yx, track_logits, radius=8):
  """Computes the certainty of the tracker."""
  shape = coord_yx.shape[:-1]
  coord_yx = coord_yx.flatten(0, -2)
  track_logits = track_logits.flatten(0, -2)
  # track_logits.shape == [b, 512]
  # coord_yx.shape == [b, 2]
  logits_y, logits_x = track_logits.chunk(2, dim=-1)
  track_softmax_y = F.softmax(logits_y, dim=-1)
  track_softmax_x = F.softmax(logits_x, dim=-1)
  sm_y, coord_y = get_window(coord_yx[:, 0:1], track_softmax_y)
  sm_x, coord_x = get_window(coord_yx[:, 1:2], track_softmax_x)
  sm = sm_y[..., :, None] * sm_x[..., None, :]
  grid_x, grid_y = torch.vmap(torch.meshgrid)(coord_x, coord_y)
  # grid_x.shape == [b, N, N]
  grid = torch.stack([grid_y, grid_x], dim=-1)
  in_radius = ((grid - coord_yx[:, None, None]) ** 2).sum(-1) <= (
      (radius**2) + 1e-8
  )
  return (sm * in_radius).sum(-1).sum(-1).reshape(*shape, 1)


def restore_model_from_jax_checkpoint(model, ckpt_path):
  """Restores a TAPNext model from a JAX checkpoint."""
  ckpt = {k: v for k, v in np.load(ckpt_path).items()}
  model.lin_proj.weight.data.copy_(
      torch.tensor(ckpt['backbone/embedding/kernel'][0]).permute(3, 2, 0, 1)
  )
  model.lin_proj.bias.data.copy_(torch.tensor(ckpt['backbone/embedding/bias']))
  model.mask_token.data.copy_(torch.tensor(ckpt['backbone/mask_token']))
  model.point_query_token.data.copy_(
      torch.tensor(ckpt['backbone/point_query_token'])
  )
  model.unknown_token.data.copy_(torch.tensor(ckpt['backbone/unknown_token']))
  model.image_pos_emb.data.copy_(torch.tensor(ckpt['backbone/pos_embedding']))
  model.encoder_norm.weight.data.copy_(
      torch.tensor(ckpt['backbone/Transformer/encoder_norm/scale'])
  )
  model.encoder_norm.bias.data.copy_(
      torch.tensor(ckpt['backbone/Transformer/encoder_norm/bias'])
  )
  for layer in range(12):
    # convert ssm part
    prefix = f'backbone/Transformer/encoderblock_{layer}/ssm_block'
    ssm_params = {
        key: torch.tensor(
            ckpt[
                f'{prefix}/'
                + re.sub('weight', 'kernel', re.sub(r'\.', '/', key))
            ]
        )
        for key, _ in model.blocks[layer].ssm_block.named_parameters()
    }
    for key in ssm_params:
      if 'weight' in key:
        ssm_params[key] = ssm_params[key].T
    model.blocks[layer].ssm_block.load_state_dict(ssm_params)

    # convert vit part
    vit_params = {
        re.sub(
            f'backbone/Transformer/encoderblock_{layer}/vit_block/', '', k
        ): v
        for k, v in ckpt.items()
        if f'backbone/Transformer/encoderblock_{layer}/vit_block' in k
    }
    torch_vit_params = {}
    torch_vit_params['ln_1.weight'] = vit_params['LayerNorm_0/scale']
    torch_vit_params['ln_1.bias'] = vit_params['LayerNorm_0/bias']
    torch_vit_params['ln_2.weight'] = vit_params['LayerNorm_1/scale']
    torch_vit_params['ln_2.bias'] = vit_params['LayerNorm_1/bias']
    torch_vit_params['mlp.0.weight'] = vit_params['MlpBlock_0/Dense_0/kernel'].T
    torch_vit_params['mlp.0.bias'] = vit_params['MlpBlock_0/Dense_0/bias']
    torch_vit_params['mlp.3.weight'] = vit_params['MlpBlock_0/Dense_1/kernel'].T
    torch_vit_params['mlp.3.bias'] = vit_params['MlpBlock_0/Dense_1/bias']
    torch_vit_params['self_attention.in_proj_weight'] = np.concatenate(
        [
            vit_params['MultiHeadDotProductAttention_0/query/kernel']
            .reshape(768, 768)
            .T,
            vit_params['MultiHeadDotProductAttention_0/key/kernel']
            .reshape(768, 768)
            .T,
            vit_params['MultiHeadDotProductAttention_0/value/kernel']
            .reshape(768, 768)
            .T,
        ],
        axis=0,
    )
    torch_vit_params['self_attention.in_proj_bias'] = np.concatenate([
        vit_params['MultiHeadDotProductAttention_0/query/bias'].flatten(),
        vit_params['MultiHeadDotProductAttention_0/key/bias'].flatten(),
        vit_params['MultiHeadDotProductAttention_0/value/bias'].flatten(),
    ])
    torch_vit_params['self_attention.out_proj.weight'] = (
        vit_params['MultiHeadDotProductAttention_0/out/kernel']
        .reshape(768, 768)
        .T
    )
    torch_vit_params['self_attention.out_proj.bias'] = vit_params[
        'MultiHeadDotProductAttention_0/out/bias'
    ].flatten()
    for k in torch_vit_params:
      torch_vit_params[k] = torch.tensor(np.array(torch_vit_params[k]))
    model.blocks[layer].vit_block.load_state_dict(torch_vit_params)
  model.visible_head[0].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_0/kernel'].T)
  )
  model.visible_head[0].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_0/bias'])
  )
  model.visible_head[1].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_1/scale'])
  )
  model.visible_head[1].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_1/bias'])
  )
  model.visible_head[3].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_3/kernel'].T)
  )
  model.visible_head[3].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_3/bias'])
  )
  model.visible_head[4].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_4/scale'])
  )
  model.visible_head[4].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_4/bias'])
  )
  model.visible_head[6].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_6/kernel'].T)
  )
  model.visible_head[6].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_6/bias'])
  )

  model.coordinate_head[0].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_0/kernel'].T)
  )
  model.coordinate_head[0].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_0/bias'])
  )
  model.coordinate_head[1].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_1/scale'])
  )
  model.coordinate_head[1].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_1/bias'])
  )
  model.coordinate_head[3].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_3/kernel'].T)
  )
  model.coordinate_head[3].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_3/bias'])
  )
  model.coordinate_head[4].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_4/scale'])
  )
  model.coordinate_head[4].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_4/bias'])
  )
  model.coordinate_head[6].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_6/kernel'].T)
  )
  model.coordinate_head[6].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_6/bias'])
  )
  return model

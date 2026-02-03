# Copyright 2026 Google LLC
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


def convert_pytorch_model_to_jax_checkpoint(model):
  """Converts a TAPNext PyTorch model back to a JAX/Numpy checkpoint dictionary.

  matching specific multi-head attention and embedding shapes.

  Args:
    model: The TAPNext PyTorch model to convert.

  Returns:
    A dictionary of JAX/Numpy checkpoint data.
  """
  ckpt = {}

  def to_np(tensor):
    """Helper to convert PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()

  # Constants inferred from shapes provided
  n_heads = 12
  head_dim = 64
  embed_dim = 768  # 12 * 64

  # --- 1. Global / Backbone Embeddings ---

  # Embedding: PyTorch (Out, In, H, W) -> JAX (1, H, W, In, Out)
  # 1. Permute PyTorch (768, 3, 8, 8) -> (8, 8, 3, 768)
  # 2. Add singleton dimension at front -> (1, 8, 8, 3, 768)
  embed_kernel = model.lin_proj.weight.permute(2, 3, 1, 0)
  ckpt['backbone/embedding/kernel'] = np.expand_dims(
      to_np(embed_kernel), axis=0
  )

  ckpt['backbone/embedding/bias'] = to_np(model.lin_proj.bias)
  ckpt['backbone/mask_token'] = to_np(model.mask_token)
  ckpt['backbone/point_query_token'] = to_np(model.point_query_token)
  ckpt['backbone/unknown_token'] = to_np(model.unknown_token)
  ckpt['backbone/pos_embedding'] = to_np(model.image_pos_emb)

  ckpt['backbone/Transformer/encoder_norm/scale'] = to_np(
      model.encoder_norm.weight
  )
  ckpt['backbone/Transformer/encoder_norm/bias'] = to_np(
      model.encoder_norm.bias
  )

  # --- 2. Transformer Blocks (Layers 0-11) ---
  for layer in range(12):
    block_prefix = f'backbone/Transformer/encoderblock_{layer}'

    # --- A. SSM Block ---
    ssm_block = model.blocks[layer].ssm_block
    for name, param in ssm_block.named_parameters():
      jax_key_suffix = name.replace('.', '/').replace('weight', 'kernel')
      jax_key = f'{block_prefix}/ssm_block/{jax_key_suffix}'

      param_np = to_np(param)
      if 'weight' in name:
        param_np = param_np.T  # Transpose linear weights

      ckpt[jax_key] = param_np

    # --- B. ViT Block ---
    vit_block = model.blocks[layer].vit_block
    vit_prefix = f'{block_prefix}/vit_block'

    # LayerNorms
    ckpt[f'{vit_prefix}/LayerNorm_0/scale'] = to_np(vit_block.ln_1.weight)
    ckpt[f'{vit_prefix}/LayerNorm_0/bias'] = to_np(vit_block.ln_1.bias)
    ckpt[f'{vit_prefix}/LayerNorm_1/scale'] = to_np(vit_block.ln_2.weight)
    ckpt[f'{vit_prefix}/LayerNorm_1/bias'] = to_np(vit_block.ln_2.bias)

    # MLP (Simple Transpose)
    ckpt[f'{vit_prefix}/MlpBlock_0/Dense_0/kernel'] = to_np(
        vit_block.mlp[0].weight
    ).T
    ckpt[f'{vit_prefix}/MlpBlock_0/Dense_0/bias'] = to_np(vit_block.mlp[0].bias)
    ckpt[f'{vit_prefix}/MlpBlock_0/Dense_1/kernel'] = to_np(
        vit_block.mlp[3].weight
    ).T
    ckpt[f'{vit_prefix}/MlpBlock_0/Dense_1/bias'] = to_np(vit_block.mlp[3].bias)

    # --- Self Attention (Reshaping needed) ---
    # PyTorch in_proj_weight is (3*Embed, Embed).
    # We chunk it into Q, K, V -> (Embed, Embed).
    q_w, k_w, v_w = torch.chunk(
        vit_block.self_attention.in_proj_weight, 3, dim=0
    )
    q_b, k_b, v_b = torch.chunk(vit_block.self_attention.in_proj_bias, 3, dim=0)

    # Helper to reshape (Embed, Embed) -> (Embed, Heads, HeadDim)
    def reshape_attn_kernel(tensor):
      # 1. Transpose: (Out, In) -> (In, Out) = (768, 768)
      # 2. Reshape: (768, 12, 64)
      return to_np(tensor).T.reshape(embed_dim, n_heads, head_dim)

    # Helper to reshape Bias (Embed,) -> (Heads, HeadDim)
    def reshape_attn_bias(tensor):
      return to_np(tensor).reshape(n_heads, head_dim)

    # Q, K, V Weights & Biases
    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/query/kernel'] = (
        reshape_attn_kernel(q_w)
    )
    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/query/bias'] = (
        reshape_attn_bias(q_b)
    )

    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/key/kernel'] = (
        reshape_attn_kernel(k_w)
    )
    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/key/bias'] = (
        reshape_attn_bias(k_b)
    )

    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/value/kernel'] = (
        reshape_attn_kernel(v_w)
    )
    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/value/bias'] = (
        reshape_attn_bias(v_b)
    )

    # Output Projection
    # PyTorch: (Embed, Embed) -> Transpose -> (Embed, Embed) -> Reshape ->
    # (Heads, HeadDim, Embed)
    # Target Shape: (12, 64, 768)
    out_proj_weight = to_np(
        vit_block.self_attention.out_proj.weight
    ).T  # (In, Out) -> (768, 768)
    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/out/kernel'] = (
        out_proj_weight.reshape(n_heads, head_dim, embed_dim)
    )

    ckpt[f'{vit_prefix}/MultiHeadDotProductAttention_0/out/bias'] = to_np(
        vit_block.self_attention.out_proj.bias
    )

  # --- 3. Heads (Visible & Coordinate) ---
  head_configs = [
      ('visible_head', model.visible_head),
      ('coordinate_head', model.coordinate_head),
  ]

  for jax_prefix, torch_module in head_configs:
    ckpt[f'{jax_prefix}/layers_0/kernel'] = to_np(torch_module[0].weight).T
    ckpt[f'{jax_prefix}/layers_0/bias'] = to_np(torch_module[0].bias)
    ckpt[f'{jax_prefix}/layers_1/scale'] = to_np(torch_module[1].weight)
    ckpt[f'{jax_prefix}/layers_1/bias'] = to_np(torch_module[1].bias)
    ckpt[f'{jax_prefix}/layers_3/kernel'] = to_np(torch_module[3].weight).T
    ckpt[f'{jax_prefix}/layers_3/bias'] = to_np(torch_module[3].bias)
    ckpt[f'{jax_prefix}/layers_4/scale'] = to_np(torch_module[4].weight)
    ckpt[f'{jax_prefix}/layers_4/bias'] = to_np(torch_module[4].bias)
    ckpt[f'{jax_prefix}/layers_6/kernel'] = to_np(torch_module[6].weight).T
    ckpt[f'{jax_prefix}/layers_6/bias'] = to_np(torch_module[6].bias)

  return ckpt

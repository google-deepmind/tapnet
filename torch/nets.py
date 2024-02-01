# Copyright 2024 DeepMind Technologies Limited
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

"""Pytorch neural network definitions."""

from typing import Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F


class ConvChannelsMixer(nn.Module):
  """Linear activation block for PIPs's MLP Mixer."""

  def __init__(self, in_channels):
    super().__init__()
    self.mlp2_up = nn.Linear(in_channels, in_channels * 4)
    self.mlp2_down = nn.Linear(in_channels * 4, in_channels)

  def forward(self, x):
    x = self.mlp2_up(x)
    x = F.gelu(x, approximate='tanh')
    x = self.mlp2_down(x)
    return x


class PIPsConvBlock(nn.Module):
  """Convolutional block for PIPs's MLP Mixer."""

  def __init__(self, in_channels, kernel_shape=3):
    super().__init__()
    self.layer_norm = nn.LayerNorm(
        normalized_shape=in_channels, elementwise_affine=True, bias=False
    )
    self.mlp1_up = nn.Conv1d(
        in_channels, in_channels * 4, kernel_shape, 1, 1, groups=in_channels
    )
    self.mlp1_up_1 = nn.Conv1d(
        in_channels * 4,
        in_channels * 4,
        kernel_shape,
        1,
        1,
        groups=in_channels * 4,
    )
    self.layer_norm_1 = nn.LayerNorm(
        normalized_shape=in_channels, elementwise_affine=True, bias=False
    )
    self.conv_channels_mixer = ConvChannelsMixer(in_channels)

  def forward(self, x):
    to_skip = x
    x = self.layer_norm(x)

    x = x.permute(0, 2, 1)
    x = self.mlp1_up(x)
    x = F.gelu(x, approximate='tanh')
    x = self.mlp1_up_1(x)
    x = x.permute(0, 2, 1)
    x = x[..., 0::4] + x[..., 1::4] + x[..., 2::4] + x[..., 3::4]

    x = x + to_skip
    to_skip = x
    x = self.layer_norm_1(x)
    x = self.conv_channels_mixer(x)

    x = x + to_skip
    return x


class PIPSMLPMixer(nn.Module):
  """Depthwise-conv version of PIPs's MLP Mixer."""

  def __init__(
      self,
      input_channels: int,
      output_channels: int,
      hidden_dim: int = 512,
      num_blocks: int = 12,
      kernel_shape: int = 3,
  ):
    """Inits Mixer module.

    A depthwise-convolutional version of a MLP Mixer for processing images.

    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults
          to 512.
        num_blocks (int, optional): The number of convolution blocks in the
          mixer. Defaults to 12.
        kernel_shape (int, optional): The size of the kernel in the convolution
          blocks. Defaults to 3.
    """

    super().__init__()
    self.hidden_dim = hidden_dim
    self.num_blocks = num_blocks
    self.linear = nn.Linear(input_channels, self.hidden_dim)
    self.layer_norm = nn.LayerNorm(
        normalized_shape=hidden_dim, elementwise_affine=True, bias=False
    )
    self.linear_1 = nn.Linear(hidden_dim, output_channels)
    self.blocks = nn.ModuleList([
        PIPsConvBlock(hidden_dim, kernel_shape) for _ in range(num_blocks)
    ])

  def forward(self, x):
    x = self.linear(x)
    for block in self.blocks:
      x = block(x)

    x = self.layer_norm(x)
    x = self.linear_1(x)
    return x


class BlockV2(nn.Module):
  """ResNet V2 block."""

  def __init__(
      self,
      channels_in: int,
      channels_out: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
  ):
    super().__init__()
    self.padding = (1, 1, 1, 1)
    # Handle assymetric padding created by padding="SAME" in JAX/LAX
    if stride == 1:
      self.padding = (1, 1, 1, 1)
    elif stride == 2:
      self.padding = (0, 2, 0, 2)
    else:
      raise ValueError(
          'Check correct padding using padtype_to_padsin jax._src.lax.lax'
      )

    self.use_projection = use_projection
    if self.use_projection:
      self.proj_conv = nn.Conv2d(
          in_channels=channels_in,
          out_channels=channels_out,
          kernel_size=1,
          stride=stride,
          padding=0,
          bias=False,
      )

    self.bn_0 = nn.InstanceNorm2d(
        num_features=channels_in,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    )
    self.conv_0 = nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=0,
        bias=False,
    )

    self.conv_1 = nn.Conv2d(
        in_channels=channels_out,
        out_channels=channels_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    self.bn_1 = nn.InstanceNorm2d(
        num_features=channels_out,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    )

  def forward(self, inputs):
    x = shortcut = inputs

    x = self.bn_0(x)
    x = torch.relu(x)
    if self.use_projection:
      shortcut = self.proj_conv(x)

    x = self.conv_0(F.pad(x, self.padding))

    x = self.bn_1(x)
    x = torch.relu(x)
    # no issues with padding here as this layer always has stride 1
    x = self.conv_1(x)

    return x + shortcut


class BlockGroup(nn.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels_in: int,
      channels_out: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
  ):
    super().__init__()
    blocks = []
    for i in range(num_blocks):
      blocks.append(
          BlockV2(
              channels_in=channels_in if i == 0 else channels_out,
              channels_out=channels_out,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
          )
      )
    self.blocks = nn.ModuleList(blocks)

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out


class ResNet(nn.Module):
  """ResNet model."""

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      channels_per_group: Sequence[int] = (64, 128, 256, 512),
      use_projection: Sequence[bool] = (True, True, True, True),
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Initializes a ResNet model with customizable layers and configurations.

    This constructor allows defining the architecture of a ResNet model by
    setting the number of blocks, channels, projection usage, and strides for
    each group of blocks within the network. It provides flexibility in
    creating various ResNet configurations.

    Args:
      blocks_per_group: A sequence of 4 integers, each indicating the number
        of residual blocks in each group.
      channels_per_group: A sequence of 4 integers, each specifying the number
        of output channels for the blocks in each group. Defaults to (64, 128,
        256, 512).
      use_projection: A sequence of 4 booleans, each indicating whether to use
        a projection shortcut (True) or an identity shortcut (False) in each
        group. Defaults to (True, True, True, True).
      strides: A sequence of 4 integers, each specifying the stride size for
        the convolutions in each group. Defaults to (1, 2, 2, 2).

    The ResNet model created will have 4 groups, with each group's
    architecture defined by the corresponding elements in these sequences.
    """
    super().__init__()

    self.initial_conv = nn.Conv2d(
        in_channels=3,
        out_channels=channels_per_group[0],
        kernel_size=(7, 7),
        stride=2,
        padding=0,
        bias=False,
    )

    block_groups = []
    for i, _ in enumerate(strides):
      block_groups.append(
          BlockGroup(
              channels_in=channels_per_group[i - 1] if i > 0 else 64,
              channels_out=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              use_projection=use_projection[i],
          )
      )
    self.block_groups = nn.ModuleList(block_groups)

  def forward(self, inputs):
    result = {}
    out = inputs
    out = self.initial_conv(F.pad(out, (2, 4, 2, 4)))
    result['initial_conv'] = out

    for block_id, block_group in enumerate(self.block_groups):
      out = block_group(out)
      result[f'resnet_unit_{block_id}'] = out

    return result

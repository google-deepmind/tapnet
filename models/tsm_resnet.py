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

"""Temporal Shift Module w/ ResNet-50 and ResNet-101.

Based on:
  TSM: Temporal Shift Module for Efficient Video Understanding
  Ji Lin, Chuang Gan, Song Han
  https://arxiv.org/pdf/1811.08383.pdf.
"""

from typing import Optional, Sequence, Union
from absl import logging

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import typing_extensions

from tapnet.models import tsm_utils as tsmu


class NormalizeFn(typing_extensions.Protocol):

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    pass


class TSMResNetBlock(hk.Module):
  """A ResNet subblock with Temporal Channel Shifting.

  Combines a typical ResNetV2 block implementation
  (see https://arxiv.org/abs/1512.03385) with a pre-convolution Temporal
  Shift Module (see https://arxiv.org/pdf/1811.08383.pdf) in the residual.
  """

  def __init__(
      self,
      output_channels: int,
      stride: int,
      use_projection: bool,
      tsm_mode: str,
      normalize_fn: Optional[NormalizeFn] = None,
      channel_shift_fraction: float = 0.125,
      num_frames: int = 8,
      rate: int = 1,
      use_bottleneck: bool = False,
      name: str = 'TSMResNetBlock',
  ):
    """Initializes the TSMResNetBlock module.

    Args:
      output_channels: Number of output channels.
      stride: Stride used in convolutions.
      use_projection: Whether to use a projection for the shortcut.
      tsm_mode: Mode for TSM ('gpu' or 'tpu' or 'deflated_0.x').
      normalize_fn: Function used for normalization.
      channel_shift_fraction: The fraction of temporally shifted channels. If
        `channel_shift_fraction` is 0, the block is the same as a normal ResNet
        block.
      num_frames: Size of frame dimension in a single batch example.
      rate: dilation rate.
      use_bottleneck: use a bottleneck (resnet-50 and above),
      name: The name of the module.
    """
    super().__init__(name=name)
    self._output_channels = (
        output_channels if use_bottleneck else output_channels // 4)
    self._bottleneck_channels = output_channels // 4
    self._stride = stride
    self._rate = rate
    self._use_projection = use_projection
    self._normalize_fn = normalize_fn
    self._tsm_mode = tsm_mode
    self._channel_shift_fraction = channel_shift_fraction
    self._num_frames = num_frames
    self._use_bottleneck = use_bottleneck

  def __call__(
      self,
      inputs: chex.Array,
      is_training: bool = True,
  ) -> chex.Array:
    """Connects the ResNetBlock module into the graph.

    Args:
      inputs: A 4-D float array of shape `[B, H, W, C]`.
      is_training: Whether to use training mode.

    Returns:
      A 4-D float array of shape
      `[B * num_frames, new_h, new_w, output_channels]`.
    """
    # ResNet V2 uses pre-activation, where the batch norm and relu are before
    # convolutions, rather than after as in ResNet V1.
    preact = inputs
    if self._normalize_fn is not None:
      preact = self._normalize_fn(preact, is_training=is_training)
    preact = jax.nn.relu(preact)

    if self._use_projection:
      shortcut = hk.Conv2D(
          output_channels=self._output_channels,
          kernel_shape=1,
          stride=self._stride,
          with_bias=False,
          padding='SAME',
          name='shortcut_conv',
      )(preact)
    else:
      shortcut = inputs

    # Eventually applies Temporal Shift Module.
    if self._channel_shift_fraction != 0:
      preact = tsmu.apply_temporal_shift(
          preact,
          tsm_mode=self._tsm_mode,
          num_frames=self._num_frames,
          channel_shift_fraction=self._channel_shift_fraction)

    # First convolution.
    residual = hk.Conv2D(
        self._bottleneck_channels,
        kernel_shape=1 if self._use_bottleneck else 3,
        stride=1 if self._use_bottleneck else self._stride,
        with_bias=False,
        padding='SAME',
        name='conv_0',
    )(preact)

    if self._use_bottleneck:
      # Second convolution.
      if self._normalize_fn is not None:
        residual = self._normalize_fn(residual, is_training=is_training)
      residual = jax.nn.relu(residual)
      residual = hk.Conv2D(
          output_channels=self._bottleneck_channels,
          kernel_shape=3,
          stride=self._stride,
          rate=self._rate,
          with_bias=False,
          padding='SAME',
          name='conv_1',
      )(residual)

    # Third convolution.
    if self._normalize_fn is not None:
      residual = self._normalize_fn(residual, is_training=is_training)
    residual = jax.nn.relu(residual)
    residual = hk.Conv2D(
        output_channels=self._output_channels,
        kernel_shape=1 if self._use_bottleneck else 3,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_2',
    )(residual)

    # NOTE: we do not use block multiplier.
    output = shortcut + residual
    return output


class TSMResNetUnit(hk.Module):
  """Block group for TSM ResNet."""

  def __init__(
      self,
      output_channels: int,
      num_blocks: int,
      stride: int,
      tsm_mode: str,
      num_frames: int,
      normalize_fn: Optional[NormalizeFn] = None,
      channel_shift_fraction: float = 0.125,
      rate: int = 1,
      use_bottleneck: bool = False,
      name: str = 'tsm_resnet_unit',
  ):
    """Creates a TSMResNet Unit.

    Args:
      output_channels: Number of output channels.
      num_blocks: Number of ResNet blocks in the unit.
      stride: Stride of the unit.
      tsm_mode: Which temporal shift module to use.
      num_frames: Size of frame dimension in a single batch example.
      normalize_fn: Function used for normalization.
      channel_shift_fraction: The fraction of temporally shifted channels. If
        `channel_shift_fraction` is 0, the block is the same as a normal ResNet
        block.
      rate: dilation rate.
      use_bottleneck: use a bottleneck (resnet-50 and above).
      name: The name of the module.
    """
    super().__init__(name=name)
    self._output_channels = output_channels
    self._num_blocks = num_blocks
    self._normalize_fn = normalize_fn
    self._stride = stride
    self._tsm_mode = tsm_mode
    self._channel_shift_fraction = channel_shift_fraction
    self._num_frames = num_frames
    self._rate = rate
    self._use_bottleneck = use_bottleneck

  def __call__(
      self,
      inputs: chex.Array,
      is_training: bool,
  ) -> chex.Array:
    """Connects the module to inputs.

    Args:
      inputs: A 4-D float array of shape `[B * num_frames, H, W, C]`.
      is_training: Whether to use training mode.

    Returns:
      A 4-D float array of shape
      `[B * num_frames, H // stride, W // stride, output_channels]`.
    """
    net = inputs
    for idx_block in range(self._num_blocks):
      net = TSMResNetBlock(
          self._output_channels,
          stride=(self._stride if idx_block == 0 else 1),
          rate=(max(self._rate // 2, 1) if idx_block == 0 else self._rate),
          use_projection=(idx_block == 0),
          normalize_fn=self._normalize_fn,
          tsm_mode=self._tsm_mode,
          channel_shift_fraction=self._channel_shift_fraction,
          num_frames=self._num_frames,
          name=f'block_{idx_block}',
      )(net, is_training=is_training)
    return net


class TSMResNetV2(hk.Module):
  """TSM based on ResNet V2 as described in https://arxiv.org/abs/1603.05027."""

  # Endpoints of the model in order.
  VALID_ENDPOINTS = (
      'tsm_resnet_stem',
      'tsm_resnet_unit_0',
      'tsm_resnet_unit_1',
      'tsm_resnet_unit_2',
      'tsm_resnet_unit_3',
      'last_conv',
      'Embeddings',
  )

  def __init__(
      self,
      normalize_fn: Optional[NormalizeFn] = None,
      depth: int = 18,
      num_frames: int = 16,
      channel_shift_fraction: Union[float, Sequence[float]] = 0.125,
      width_mult: int = 1,
      name: str = 'TSMResNetV2',
  ):
    """Constructs a ResNet model.

    Args:
      normalize_fn: Function used for normalization.
      depth: Depth of the desired ResNet.
      num_frames: Number of frames (used in TPU mode).
      channel_shift_fraction: Fraction of channels that are temporally shifted,
        if `channel_shift_fraction` is 0, a regular ResNet is returned.
      width_mult: Whether or not to use a width multiplier.
      name: The name of the module.

    Raises:
      ValueError: If `channel_shift_fraction` or `depth` has invalid value.
    """
    super().__init__(name=name)

    if isinstance(channel_shift_fraction, float):
      channel_shift_fraction = [channel_shift_fraction] * 4

    if not all([0. <= x <= 1.0 for x in channel_shift_fraction]):
      raise ValueError(f'channel_shift_fraction ({channel_shift_fraction})'
                       ' all have to be in [0, 1].')

    self._num_frames = num_frames

    self._channels = (256, 512, 1024, 2048)

    num_blocks = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3),
    }
    if depth not in num_blocks:
      raise ValueError(
          f'`depth` should be in {list(num_blocks.keys())} ({depth} given).')
    self._num_blocks = num_blocks[depth]

    self._width_mult = width_mult
    self._channel_shift_fraction = channel_shift_fraction
    self._normalize_fn = normalize_fn
    self._use_bottleneck = (depth >= 50)

  def __call__(
      self,
      inputs: chex.Array,
      is_training: bool = True,
      final_endpoint: str = 'Embeddings',
      is_deflated: bool = False,
      alpha_deflation: float = 0.3,
      out_num_frames: Optional[int] = None,
      output_stride: int = 8,
  ) -> chex.Array:
    """Connects the TSM ResNetV2 module into the graph.

    Args:
      inputs: The input may be in one of two shapes; if the shape is `[B, T, H,
        W, C]`, this module assumes the backend is a GPU (setting
        `tsm_mode='gpu'`) and `T` is treated the time dimension, with `B` being
        the batch dimension. This mode cannot be used when `is_deflated` is
        `true`. In this mode, the num_frames parameter passed to the constructor
        is ignored. If the shape is `[B, H, W, C]`, then the batch dimension is
        assumed to be of the form [B*T, H, W, C], where `T` is the number of
        frames in each video. This value may be set by passing `num_frames=n` to
        the constructor. The default value is `n=16` (beware this default is not
        the same as the default for the `TSMResNetBlock`, which has a default of
        8 frames). In this case, the module assumes it is being run on a TPU,
        and emits instructions that are more efficient for that case,
        using`tsm_mode`='tpu'` for the downstream blocks.
      is_training: Whether to use training mode.
      final_endpoint: Up to which endpoint to run / return.
      is_deflated: Whether or not to use the deflated version of the network.
      alpha_deflation: Deflation parameter to use for dealing with the padding
        effect.
      out_num_frames: Whether time is on first axis, for TPU performance
      output_stride: Stride of the final feature grid; possible values are
        4, 8, 16, or 32.  32 is the standard for TSM-ResNet. Others strides are
        achieved by converting strided to un-strided convolutions later in the
        network, while increasing the dilation rate for later layers.

    Returns:
      Network output at location `final_endpoint`. A float array which shape
      depends on `final_endpoint`.

    Raises:
      ValueError: If `final_endpoint` is not recognized.
    """

    # Prepare inputs for TSM.
    if is_deflated:
      if len(inputs.shape) != 4:
        raise ValueError(
            'In deflated mode inputs should be given as [B, H, W, 3]')
      logging.warning(
          'Deflation is an experimental feature and the API might change.')
      tsm_mode = f'deflated_{alpha_deflation}'
      num_frames = 1
    else:
      inputs, tsm_mode, num_frames = tsmu.prepare_inputs(inputs)
      num_frames = num_frames or out_num_frames or self._num_frames

    self._final_endpoint = final_endpoint
    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError(f'Unknown final endpoint {self._final_endpoint}')

    # Stem convolution.
    end_point = 'tsm_resnet_stem'
    net = hk.Conv2D(
        output_channels=64 * self._width_mult,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        name=end_point,
        padding='SAME',
    )(inputs)
    net = hk.MaxPool(
        window_shape=(1, 3, 3, 1),
        strides=(1, 2, 2, 1),
        padding='SAME',
    )(net)
    if self._final_endpoint == end_point:
      net = tsmu.prepare_outputs(net, tsm_mode, num_frames, reduce_mean=False)
      return net

    if output_stride == 4:
      strides = (1, 1, 1, 1)
      rates = (1, 2, 4, 8)
    elif output_stride == 8:
      strides = (1, 2, 1, 1)
      rates = (1, 1, 2, 4)
    elif output_stride == 16:
      strides = (1, 2, 2, 1)
      rates = (1, 1, 1, 2)
    elif output_stride == 32:
      strides = (1, 2, 2, 2)
      rates = (1, 1, 1, 1)
    else:
      raise ValueError('unsupported output stride')

    # Residual block.
    for unit_id, (channels, num_blocks, stride, rate) in enumerate(
        zip(self._channels, self._num_blocks, strides, rates)):
      end_point = f'tsm_resnet_unit_{unit_id}'
      net = TSMResNetUnit(
          output_channels=channels * self._width_mult,
          num_blocks=num_blocks,
          stride=stride,
          rate=rate,
          normalize_fn=self._normalize_fn,
          channel_shift_fraction=self._channel_shift_fraction[unit_id],
          num_frames=num_frames,
          tsm_mode=tsm_mode,
          use_bottleneck=self._use_bottleneck,
          name=end_point,
      )(net, is_training=is_training)
      if self._final_endpoint == end_point:
        net = tsmu.prepare_outputs(net, tsm_mode, num_frames, reduce_mean=False)
        return net

    if self._normalize_fn is not None:
      net = self._normalize_fn(net, is_training=is_training)
    net = jax.nn.relu(net)

    end_point = 'last_conv'
    if self._final_endpoint == end_point:
      net = tsmu.prepare_outputs(net, tsm_mode, num_frames, reduce_mean=False)
      return net
    net = jnp.mean(net, axis=(1, 2))
    # Prepare embedding outputs for TSM (temporal average of features).
    net = tsmu.prepare_outputs(net, tsm_mode, num_frames, reduce_mean=True)
    assert self._final_endpoint == 'Embeddings'
    return net

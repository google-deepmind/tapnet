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

"""ResNet with BatchNorm, LayerNorm, InstanceNorm or None."""

from typing import Mapping, Optional, Sequence, Union

import haiku as hk
import jax


FloatStrOrBool = Union[str, float, bool]


class BlockV1(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      normalization: Optional[str] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.normalization = normalization

    if normalization == "batchnorm":
      bn_config = dict(bn_config)
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)
      bn_config.setdefault("decay_rate", 0.9)
      bn_config.setdefault("cross_replica_axis", "i")
    elif normalization == "layernorm":
      bn_config = dict(bn_config)
      bn_config.setdefault("axis", [-1, -2, -3])
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)
    elif normalization == "instancenorm":
      bn_config = dict(bn_config)
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

      if normalization == "batchnorm":
        self.proj_batchnorm = hk.BatchNorm(
            name="shortcut_batchnorm", **bn_config
        )
      elif normalization == "layernorm":
        self.proj_norm = hk.LayerNorm(name="shortcut_layernorm", **bn_config)
      elif normalization == "instancenorm":
        self.proj_norm = hk.InstanceNorm(
            name="shortcut_instancenorm", **bn_config
        )

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")
    if normalization == "batchnorm":
      bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)
    elif normalization == "layernorm":
      bn_0 = hk.LayerNorm(name="layernorm_0", **bn_config)
    elif normalization == "instancenorm":
      bn_0 = hk.InstanceNorm(name="instancenorm_0", **bn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    if normalization == "batchnorm":
      bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
    elif normalization == "layernorm":
      bn_1 = hk.LayerNorm(name="layernorm_1", **bn_config)
    elif normalization == "instancenorm":
      bn_1 = hk.InstanceNorm(name="instancenorm_1", **bn_config)
    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      if normalization == "batchnorm":
        bn_2 = hk.BatchNorm(name="batchnorm_2", **bn_config)
      elif normalization == "layernorm":
        bn_2 = hk.LayerNorm(name="layernorm_2", **bn_config)
      elif normalization == "instancenorm":
        bn_2 = hk.InstanceNorm(name="instancenorm_2", **bn_config)
      layers = layers + ((conv_2, bn_2),)

    self.layers = layers

  def __call__(self, inputs, is_training, test_local_stats):
    out = shortcut = inputs

    if self.use_projection:
      shortcut = self.proj_conv(shortcut)
      if self.normalization == "batchnorm":
        shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)
      elif self.normalization in ["layernorm", "instancenorm"]:
        shortcut = self.proj_norm(shortcut)

    for i, (conv_i, bn_i) in enumerate(self.layers):
      out = conv_i(out)
      if self.normalization == "batchnorm":
        out = bn_i(out, is_training, test_local_stats)
      elif self.normalization in ["layernorm", "instancenorm"]:
        out = bn_i(out)
      if i < len(self.layers) - 1:  # Don't apply relu on last layer
        out = jax.nn.relu(out)

    return jax.nn.relu(out + shortcut)


class BlockV2(hk.Module):
  """ResNet V2 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      normalization: Optional[str] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.normalization = normalization

    bn_config = dict(bn_config)
    if normalization == "batchnorm":
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)
      bn_config.setdefault("decay_rate", 0.9)
      bn_config.setdefault("cross_replica_axis", "i")
    elif normalization == "layernorm":
      bn_config.setdefault("axis", [-1, -2, -3])
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)
    elif normalization == "instancenorm":
      bn_config = dict(bn_config)
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")

    if normalization == "batchnorm":
      bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)
    elif normalization == "layernorm":
      bn_0 = hk.LayerNorm(name="layernorm_0", **bn_config)
    elif normalization == "instancenorm":
      bn_0 = hk.InstanceNorm(name="instancenorm_0", **bn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    if normalization == "batchnorm":
      bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
    elif normalization == "layernorm":
      bn_1 = hk.LayerNorm(name="layernorm_1", **bn_config)
    elif normalization == "instancenorm":
      bn_1 = hk.InstanceNorm(name="instancenorm_1", **bn_config)
    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      if normalization == "batchnorm":
        bn_2 = hk.BatchNorm(name="batchnorm_2", **bn_config)
      elif normalization == "layernorm":
        bn_2 = hk.LayerNorm(name="layernorm_2", **bn_config)
      elif normalization == "instancenorm":
        bn_2 = hk.InstanceNorm(name="instancenorm_2", **bn_config)
      layers = layers + ((conv_2, bn_2),)

    self.layers = layers

  def __call__(self, inputs, is_training, test_local_stats):
    x = shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(self.layers):
      if self.normalization == "batchnorm":
        x = bn_i(x, is_training, test_local_stats)
      elif self.normalization in ["layernorm", "instancenorm"]:
        x = bn_i(x)
      x = jax.nn.relu(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      x = conv_i(x)

    return x + shortcut


class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      bn_config: Mapping[str, FloatStrOrBool],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      normalization: Optional[str] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2 if resnet_v2 else BlockV1

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(
              channels=channels,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
              bottleneck=bottleneck,
              normalization=normalization,
              bn_config=bn_config,
              name="block_%d" % i,
          )
      )

  def __call__(self, inputs, is_training, test_local_stats):
    out = inputs
    for block in self.blocks:
      out = block(out, is_training, test_local_stats)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
  """ResNet model."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      34: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      101: {
          "blocks_per_group": (3, 4, 23, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      152: {
          "blocks_per_group": (3, 8, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      200: {
          "blocks_per_group": (3, 24, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
  }

  BlockGroup = BlockGroup  # pylint: disable=invalid-name
  BlockV1 = BlockV1  # pylint: disable=invalid-name
  BlockV2 = BlockV2  # pylint: disable=invalid-name

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      normalization: Optional[str] = "batchnorm",
      bottleneck: bool = False,
      channels_per_group: Sequence[int] = (64, 128, 256, 512),
      use_projection: Sequence[bool] = (True, True, True, True),
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      strides: Sequence[int] = (1, 2, 2, 2),
      use_max_pool: bool = True,
  ):
    """Constructs a ResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      normalization: Use `batchnorm`, `layernorm`, `instancenorm` or None.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number of
        channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride of
        convolutions for each block in each group.
      use_max_pool: Whether use max pooling after first convolution layer.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2
    self.normalization = normalization
    self.use_max_pool = use_max_pool

    if normalization == "batchnorm":
      bn_config = dict(bn_config or {})
      bn_config.setdefault("decay_rate", 0.9)
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)
      bn_config.setdefault("cross_replica_axis", "i")
    elif normalization == "layernorm":
      bn_config = dict(bn_config or {})
      bn_config.setdefault("axis", [-1, -2, -3])
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)
    elif normalization == "instancenorm":
      bn_config = dict(bn_config or {})
      bn_config.setdefault("create_scale", True)
      bn_config.setdefault("create_offset", True)

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

    initial_conv_config = dict(initial_conv_config or {})
    initial_conv_config.setdefault("output_channels", 64)
    initial_conv_config.setdefault("kernel_shape", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")

    self.initial_conv = hk.Conv2D(**initial_conv_config)

    if not self.resnet_v2 and normalization == "batchnorm":
      self.initial_batchnorm = hk.BatchNorm(
          name="initial_batchnorm", **bn_config)
    elif not self.resnet_v2 and normalization == "layernorm":
      self.initial_norm = hk.LayerNorm(name="initial_layernorm", **bn_config)
    elif not self.resnet_v2 and normalization == "instancenorm":
      self.initial_norm = hk.InstanceNorm(
          name="initial_instancenorm", **bn_config
      )

    self.block_groups = []
    for i, stride in enumerate(strides):
      self.block_groups.append(
          BlockGroup(
              channels=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=stride,
              bn_config=bn_config,
              normalization=normalization,
              resnet_v2=resnet_v2,
              bottleneck=bottleneck,
              use_projection=use_projection[i],
              name="block_group_%d" % i,
          )
      )

  def __call__(self, inputs, is_training, test_local_stats=False):
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      if self.normalization == "batchnorm":
        out = self.initial_batchnorm(out, is_training, test_local_stats)
      elif self.normalization in ["layernorm", "instancenorm"]:
        out = self.initial_norm(out)
      out = jax.nn.relu(out)

    if self.use_max_pool:
      out = hk.max_pool(
          out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
      )

    result = {}
    for block_id, block_group in enumerate(self.block_groups):
      out = block_group(out, is_training, test_local_stats)
      result[f"resnet_unit_{block_id}"] = out

    return result

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

"""Base layers."""

from collections.abc import Sequence
from typing import NamedTuple

import einops
from tapnet.tapnext.pscan import pscan
import torch
from torch import nn
from torch.nn import functional as F

_MAX_SQRT_GRADIENT = 1000.0


class RMSNorm(nn.Module):
  """RMS Norm."""

  def __init__(
      self,
      width: int,
      eps: float = 1e-6,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.eps = eps

    # Parameters.
    self.scale = nn.Parameter(
        torch.empty([self.width], device=device, dtype=dtype)
    )

  def forward(self, x):
    """Calls the RMSNorm."""
    var = torch.mean(torch.square(x), axis=-1, keepdims=True)
    normed_x = x * torch.rsqrt(var + self.eps)

    scale = torch.reshape(self.scale, [1 for _ in range(x.ndim - 1)] + [-1])

    return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer."""

  def __init__(
      self,
      width: int,
      num_blocks: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.num_blocks = num_blocks
    self.w_init_variance_scale = w_init_variance_scale
    self.block_width = self.width // self.num_blocks

    # Parameters.
    self.w = nn.Parameter(
        torch.empty(
            [self.num_blocks, self.block_width, self.block_width],
            device=device,
            dtype=dtype,
        )
    )
    self.b = nn.Parameter(
        torch.empty(
            [self.num_blocks, self.block_width], device=device, dtype=dtype
        )
    )

  def forward(self, x):
    """Calls the BlockDiagonalLinear."""
    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


def rnn_scan(x, a, h0, acc_dtype=torch.float32, use_linear_scan=True):
  """Runs the recurrence of a linear RNN.

  Uses linear scan when given 1 timestep and parallel scan when given >1
  timesteps.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    h0: The initial hidden state.
    acc_dtype: The data type for the accumulation.
    use_linear_scan: Whether to use linear scan.

  Returns:
    The output of the linear recurrence.
  """
  assert x.ndim == 3
  assert a.shape == x.shape[-a.ndim :]
  assert a.dtype == x.dtype
  assert type(a) is type(x)
  assert h0 is None or h0.dtype == acc_dtype

  if x.shape[1] == 1:
    # Using scan in sampling mode.
    if h0 is None:
      return x, x[:, 0].type(acc_dtype)
    else:
      y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
      return y.type(x.dtype), y[:, -1]
  else:
    if h0 is not None:
      h_t = h0
    else:
      h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)
    if use_linear_scan:
      y = torch.zeros_like(x)
      for t in range(x.shape[1]):
        h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
        y[:, t] = h_t.type(x.dtype)
    else:
      # Using parallel scan!
      y = pscan(a, x, h_t)
      h_t = y[:, -1]
  return y, h_t


class SqrtBoundDerivative(torch.autograd.Function):
  """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

  @staticmethod
  def forward(ctx, x: torch.Tensor) -> torch.Tensor:
    """The forward pass, which is a normal `sqrt`."""
    ctx.save_for_backward(x)
    return torch.sqrt(x)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    """The backward pass, which clips the `sqrt` gradient."""
    (x,) = ctx.saved_tensors
    clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
    return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters and layers.
    self.a_param = nn.Parameter(
        torch.empty([self.width], device=device, dtype=dtype)
    )
    self.input_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=w_init_variance_scale,
        device=device,
        dtype=dtype,
    )
    self.a_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

  def forward(self, x, cache=None, use_linear_scan=True):
    _, l, _ = x.shape
    segment_pos = torch.arange(l, device=x.device)
    if cache is not None:
      segment_pos += 1
    reset = segment_pos == 0

    # Gates for x and a.
    gate_x = torch.sigmoid(self.input_gate(x))
    gate_a = torch.sigmoid(self.a_gate(x))
    # Compute the parameter `A` of the recurrence.
    log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
    a = torch.exp(log_a)
    a_square = torch.exp(2 * log_a)
    # Gate the input.
    gated_x = x * gate_x
    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    multiplier = SqrtBoundDerivative.apply(1 - a_square)
    multiplier = reset[..., None] + ~reset[..., None] * multiplier
    normalized_x = gated_x * multiplier.type(x.dtype)

    y, last_h = rnn_scan(
        x=normalized_x, a=a, h0=cache, use_linear_scan=use_linear_scan
    )

    return y, last_h

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      width: int,
      device: str | torch.device | None = None,
  ):
    """Returns an empty initialized cache for the RG-LRU."""
    # RG-LRU cache always in float32.
    return torch.zeros((batch_size, width), dtype=torch.float32, device=device)


class CausalConv1D(nn.Module):
  """A 1D temporal convolution layer."""

  def __init__(
      self,
      width: int,
      temporal_width: int,
      w_init_variance_scale: float = 0.01,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.temporal_width = temporal_width
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters.
    self.w = nn.Parameter(
        torch.empty(
            [self.temporal_width, self.width], device=device, dtype=dtype
        )
    )
    self.b = nn.Parameter(torch.empty([width], device=device, dtype=dtype))

  def forward(self, x, cache=None):
    if cache is None:
      cache = torch.zeros(
          (x.shape[0], self.temporal_width - 1, x.shape[2]),
          dtype=x.dtype,
          device=x.device,
      )
    assert cache.shape[1] == (self.temporal_width - 1)
    x = torch.cat([cache, x], dim=1)
    one_step = x.shape[1] == self.temporal_width
    if one_step:
      y = (x * self.w.unsqueeze(0)).sum(1, keepdims=True) + self.b[
          None, None, :
      ]
    else:
      y = F.conv1d(
          x.transpose(1, 2), self.w.t().unsqueeze(1), self.b, groups=x.shape[-1]
      )
      y = y.transpose(1, 2).contiguous()
    new_cache = x[:, 1 - self.temporal_width :]
    return y, new_cache

  @classmethod
  def init_cache(
      cls,
      *,
      batch_size: int,
      width: int,
      dtype: torch.dtype,
      conv1d_temporal_width: int = 4,
      device=None
  ):
    """Returns an empty initialized cache for the Conv1D."""
    shape = (batch_size, conv1d_temporal_width - 1, width)
    return torch.zeros(shape, dtype=dtype, device=device)


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      w_shape: Sequence[int],
      b_shape: Sequence[int],
      eqn: str,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.w_shape = tuple(w_shape)
    self.b_shape = tuple(b_shape)
    self.eqn = eqn
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters.
    self.w = nn.Parameter(torch.empty(self.w_shape, device=device, dtype=dtype))
    self.b = nn.Parameter(torch.empty(self.b_shape, device=device, dtype=dtype))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Calls the Einsum."""
    return torch.einsum(self.eqn, x, self.w) + self.b


class RecurrentBlockCache(NamedTuple):
  rg_lru_state: torch.Tensor  # "*b e"
  conv1d_state: torch.Tensor  # "*b w e"


def gelu(x: torch.Tensor) -> torch.Tensor:
  """Returns the GELU activation function with the same approximation as JAX."""
  return nn.functional.gelu(x, approximate="tanh")


class RecurrentBlock(nn.Module):
  """A block that combines a linear layer, a 1D convolution, and an RG-LRU."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.lru_width = lru_width or width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.linear_y = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_x = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_out = nn.Linear(
        in_features=self.lru_width,
        out_features=self.width,
        device=device,
        dtype=dtype,
    )
    self.conv_1d = CausalConv1D(
        width=self.lru_width,
        temporal_width=self.conv1d_temporal_width,
        device=device,
        dtype=dtype,
    )
    self.rg_lru = RGLRU(
        width=self.lru_width,
        num_heads=self.num_heads,
        device=device,
        dtype=dtype,
    )

  def forward(
      self, x, cache: RecurrentBlockCache | None = None, use_linear_scan=True
  ):
    y = self.linear_y(x)
    y = gelu(y)
    x = self.linear_x(x)
    x, conv1d_state = self.conv_1d(
        x=x,
        cache=None if cache is None else cache.conv1d_state,
    )
    x, rg_lru_state = self.rg_lru(
        x=x,
        cache=None if cache is None else cache.rg_lru_state,
        use_linear_scan=use_linear_scan,
    )

    # Join branches.
    x = x * y
    x = self.linear_out(x)

    return x, RecurrentBlockCache(
        conv1d_state=conv1d_state,
        rg_lru_state=rg_lru_state,
    )

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      lru_width: int,
      dtype: torch.dtype,
      conv1d_temporal_width: int = 4,
      device: str | torch.device | None = None,
  ) -> RecurrentBlockCache:
    """Initializes an empty RG-LRU and Conv1D cache for the block."""
    return RecurrentBlockCache(
        rg_lru_state=RGLRU.init_cache(
            batch_size=batch_size,
            width=lru_width,
            device=device,
        ),
        conv1d_state=CausalConv1D.init_cache(
            batch_size=batch_size,
            width=lru_width,
            dtype=dtype,
            conv1d_temporal_width=conv1d_temporal_width,
            device=device,
        ),
    )


class MLPBlock(nn.Module):
  """A block that implements a feed-forward network with a GELU activation."""

  def __init__(
      self,
      width: int,
      expanded_width: int,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.expanded_width = expanded_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.ffw_up = Einsum(
        w_shape=(2, self.width, self.expanded_width),
        b_shape=(2, 1, 1, self.expanded_width),
        eqn="...td,cdD->c...tD",
        device=device,
        dtype=dtype,
    )
    self.ffw_down = nn.Linear(
        in_features=self.expanded_width,
        out_features=self.width,
        device=device,
        dtype=dtype,
    )

  def forward(self, x):
    out = self.ffw_up(x)
    gate_value = gelu(out[0])
    activations = gate_value * out[1]
    return self.ffw_down(activations)


class ResidualBlock(nn.Module):
  """Griffin and Hawk's residual block."""

  def __init__(
      self,
      width: int,
      mlp_expanded_width: int,
      num_heads: int,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.width = width
    self.mlp_expanded_width = mlp_expanded_width
    self.num_heads = num_heads
    self.lru_width = lru_width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Sub-blocks and layers.
    self.temporal_pre_norm = RMSNorm(
        width=self.width, device=device, dtype=dtype
    )

    self.recurrent_block = RecurrentBlock(
        width=self.width,
        num_heads=self.num_heads,
        lru_width=self.lru_width,
        conv1d_temporal_width=self.conv1d_temporal_width,
        final_w_init_variance_scale=self.final_w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

    self.channel_pre_norm = RMSNorm(
        width=width,
        device=device,
        dtype=dtype,
    )
    self.mlp_block = MLPBlock(
        width=self.width,
        expanded_width=self.mlp_expanded_width,
        final_w_init_variance_scale=self.final_w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

  def forward(
      self, x, cache: RecurrentBlockCache | None = None, use_linear_scan=True
  ):
    raw_x = x
    inputs_normalized = self.temporal_pre_norm(raw_x)
    x, cache = self.recurrent_block(inputs_normalized, cache, use_linear_scan)
    residual = x + raw_x
    x = self.channel_pre_norm(residual)
    x = self.mlp_block(x)
    x = x + residual
    return x, cache

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      width: int,
      dtype: torch.dtype,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      device: str | torch.device | None = None,
  ) -> RecurrentBlockCache:
    """Initializes an empty cache for the block."""
    return RecurrentBlock.init_cache(
        batch_size=batch_size,
        lru_width=lru_width or width,
        dtype=dtype,
        conv1d_temporal_width=conv1d_temporal_width,
        device=device,
    )

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

"""Parallel Scan Operation."""

import torch


def safe_div(numerator, denominator):
  return torch.where(
      torch.abs(denominator) < 1e-5,
      numerator * 100000.0,
      numerator / denominator,
  )


class PScan(torch.autograd.Function):
  """Implements a parallel scan operation.

  Given A is (N, T, D) and X is (N, T, D), expands A and X in-place in O(T),
  and O(log(T)) if not core-bounded, so that:
    Y[:, 0] = Y_init
    Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
  can be computed as:
    Y[:, t] = A[:, t] * Y_init + X[:, t]
  """

  @classmethod
  def expand(cls, weights, bias):
    if weights.size(1) == 1:
      return
    t_even = 2 * (weights.size(1) // 2)

    w_pairs = weights[:, :t_even].view(weights.size(0), t_even // 2, 2, -1)
    b_pairs = bias[:, :t_even].view(bias.size(0), t_even // 2, 2, -1)

    b_pairs[:, :, 1].add_(w_pairs[:, :, 1] * b_pairs[:, :, 0])
    w_pairs[:, :, 1].mul_(w_pairs[:, :, 0])

    PScan.expand(w_pairs[:, :, 1], b_pairs[:, :, 1])

    b_pairs[:, 1:, 0].add_(w_pairs[:, 1:, 0] * b_pairs[:, :-1, 1])
    w_pairs[:, 1:, 0].mul_(w_pairs[:, :-1, 1])

    if t_even < weights.size(1):
      bias[:, -1].add_(weights[:, -1] * bias[:, -2])
      weights[:, -1].mul_(weights[:, -2])

  @classmethod
  def accrev(cls, tensor):
    if tensor.size(1) == 1:
      return
    t_even = 2 * (tensor.size(1) // 2)

    pairs = tensor[:, -t_even:].view(tensor.size(0), t_even // 2, 2, -1)

    pairs[:, :, 0].add_(pairs[:, :, 1])
    PScan.accrev(pairs[:, :, 0])
    pairs[:, :-1, 1].add_(pairs[:, 1:, 0])

    if t_even < tensor.size(1):
      tensor[:, 0].add_(tensor[:, 1])

  @classmethod
  def forward(cls, ctx, weights, bias, y_init):
    ctx.weights_orig = weights.clone()
    ctx.y_init_expanded = y_init[:, None, :].clone()
    ctx.weights_expanded = weights.clone()
    ctx.bias_expanded = bias.clone()

    PScan.expand(ctx.weights_expanded, ctx.bias_expanded)
    output = ctx.weights_expanded * ctx.y_init_expanded + ctx.bias_expanded
    return output

  @classmethod
  def backward(cls, ctx, grad_output):
    grad_input_wrt_output = grad_output * ctx.weights_expanded
    grad_accumulated = grad_input_wrt_output.clone()

    PScan.accrev(grad_accumulated)

    grad_weights = safe_div(ctx.y_init_expanded, ctx.weights_orig)
    grad_weights[:, 1:].add_(
        safe_div(ctx.bias_expanded[:, :-1], ctx.weights_expanded[:, 1:])
    )

    grad_bias = safe_div(grad_accumulated, ctx.weights_expanded)
    grad_y_init = grad_input_wrt_output.sum(dim=1)

    return grad_weights * grad_accumulated, grad_bias, grad_y_init


pscan = PScan.apply

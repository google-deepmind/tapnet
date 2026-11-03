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
"""Tests for visibility-weighted TAPNext coordinate loss."""

import pytest
import torch

from tapnet.tapnext import torch_losses


def _fixture():
  logits = torch.tensor(
      [[[[4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 4.0]]]],
      requires_grad=True,
  )
  labels = torch.full((1, 1, 2, 2), 0.5)
  return logits, labels


def _reference(logits, mask):
  """Independent logsumexp/gather CE for labels at coordinate bin zero."""
  loss_x = torch.logsumexp(logits[..., :2], dim=-1) - logits[..., 0]
  loss_y = torch.logsumexp(logits[..., 2:], dim=-1) - logits[..., 2]
  return ((loss_x + loss_y) * mask).sum() / mask.sum()


@pytest.mark.parametrize(
    'visible', [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75], [1.0, 1.0]]
)
def test_coordinate_softmax_weights_each_point(visible):
  logits, labels = _fixture()
  mask = torch.tensor(visible).reshape(1, 1, 2, 1)
  got = torch_losses.coordinate_softmax(logits, labels, mask, pixel_size=2)
  expected = _reference(logits, mask.squeeze(-1))
  torch.testing.assert_close(got, expected)


def test_masked_point_does_not_receive_coordinate_gradient():
  logits, labels = _fixture()
  mask = torch.tensor([[[[1.0], [0.0]]]])
  loss = torch_losses.coordinate_softmax(logits, labels, mask, pixel_size=2)
  loss.backward()
  assert logits.grad is not None
  torch.testing.assert_close(
      logits.grad[..., 1, :], torch.zeros_like(logits.grad[..., 1, :])
  )
  assert logits.grad[..., 0, :].abs().sum() > 0


def test_only_visible_point_contributes_to_loss():
  logits, labels = _fixture()
  mask = torch.tensor([[[[1.0], [0.0]]]])
  loss = torch_losses.coordinate_softmax(logits, labels, mask, pixel_size=2)
  torch.testing.assert_close(
      loss, 2 * torch.log1p(torch.exp(torch.tensor(-4.0)))
  )


def test_plain_mask_and_trailing_singleton_mask_agree():
  logits, labels = _fixture()
  mask = torch.tensor([[[1.0, 0.0]]])
  plain = torch_losses.coordinate_softmax(
      logits, labels.clone(), mask, pixel_size=2
  )
  singleton = torch_losses.coordinate_softmax(
      logits, labels.clone(), mask.unsqueeze(-1), pixel_size=2
  )
  torch.testing.assert_close(plain, singleton)


def test_mask_must_match_coordinate_dimensions_not_broadcast():
  logits, labels = _fixture()
  with pytest.raises(ValueError, match='Visibility mask shape'):
    torch_losses.coordinate_softmax(
        logits, labels, torch.ones((1, 1, 2, 2)), pixel_size=2
    )


def test_multibatch_softmax_matches_independent_reference():
  torch.manual_seed(42)
  logits = torch.randn(2, 3, 4, 4)
  labels = torch.full((2, 3, 4, 2), 0.5)
  mask = torch.rand(2, 3, 4)
  got = torch_losses.coordinate_softmax(logits, labels, mask, pixel_size=2)
  torch.testing.assert_close(got, _reference(logits, mask))


def test_coordinate_loss_public_backward_helper_respects_visibility():
  class TinyTracker(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.logits = torch.nn.Parameter(
          torch.tensor([[[[4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 4.0]]]])
      )
      self.tracks = torch.nn.Parameter(torch.zeros((1, 1, 2, 2)))
      self.visible_logits = torch.nn.Parameter(torch.zeros((1, 1, 2, 1)))

    def forward(self, video, query_points):
      del video, query_points
      return self.tracks, self.logits, self.visible_logits, None

  tracker = TinyTracker()
  batch = {
      'video': torch.zeros((1, 1, 2, 2, 3)),
      'query_points': torch.zeros((1, 2, 3)),
      'target_points': torch.full((1, 2, 1, 2), 0.5),
      'visible': torch.tensor([[[[1.0]], [[0.0]]]]),
  }
  got = torch_losses.tapnext_loss_and_grad(tracker, batch)
  assert got > 0
  assert tracker.logits.grad is not None
  assert tracker.logits.grad[0, 0, 0].abs().sum() > 0
  torch.testing.assert_close(
      tracker.logits.grad[0, 0, 1],
      torch.zeros_like(tracker.logits.grad[0, 0, 1]),
  )

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

"""Losses for TAPNext."""

import einops
import torch
from torch.nn import functional as F


def huber_coordinate_loss(
    pred_points, target_points, mask, delta=1.0, pixel_size=256
):
  """Computes the Huber loss between predicted and target coordinates.

  Args:
    pred_points (*shape, 2): point coordinates predicted by the model
    target_points (*shape, 2): target point coordinates
    mask (*shape): visibility mask
    delta (float): the threshold of the Huber loss
    pixel_size (int): pixel size of the image

  Returns:
      Continuous huber loss (*shape)
  """
  pred_points = pred_points.float()
  target_points = target_points.float()
  target_points = target_points.clip(0, pixel_size - 1)
  error = pred_points - target_points
  error = error.clip(-1e8, 1e8)  # add magnitude bound to prevent nan
  distsqr = torch.sum(torch.square(error), dim=-1, keepdims=True)
  dist = torch.sqrt(distsqr + 1e-12)
  loss = torch.where(
      dist < delta,
      distsqr / 2,
      delta * (torch.abs(dist) - delta / 2),
  )
  mask = mask.float()
  loss = (loss * mask).sum() / mask.sum()
  return loss


def coordinate_softmax(logits, labels, mask, pixel_size=256):
  """Computes the softmax loss between predicted logits and target coordinates.

  Args:
    logits (*shape, n_bins x 2): marginal softmax logits for predicting x and y
      coordinates
    labels (*shape, 2): taget coordinates
    mask (*shape): visibility mask
    pixel_size (int): pixel size of the image

  Returns:
    loss (float): the softmax loss
  """
  logits = logits.float()
  labels = labels.float()
  labels -= 0.5
  labels = labels.clip(0, pixel_size - 1)
  labels = torch.round(labels).long()
  logits = einops.rearrange(logits, 'b ... c -> b c ...')
  labels = einops.rearrange(labels, 'b ... c -> b c ...')
  logits_x, logits_y = logits.chunk(2, dim=1)
  labels_x, labels_y = labels.chunk(2, dim=1)
  print(logits_x.shape, labels_x.shape)
  loss_x = F.cross_entropy(logits_x, labels_x.squeeze(1))
  loss_y = F.cross_entropy(logits_y, labels_y.squeeze(1))
  loss = loss_x + loss_y
  mask = mask.float()
  loss = (loss * mask).sum() / mask.sum()
  return loss


def tapnext_loss_and_grad(model, batch, loss_weight=1.0):
  """Computes the TAPNext loss and performs backward pass on the model.

  Use the init arg `use_checkpointing=True` when constructing TAPNext to
  optimize memory; this does not have any impact on the inference speed/quality.

  Args:
    model (TAPNext):
    batch (dict): a dictionary with 4 keys: * 'video' - a float32 tensor of
      shape [batch, time, height, width, 3]; it should be mean-std normalized
      (e.g. ImageNet normalization) * 'query_points' - a float32 tensor of shape
      [batch, num_queries, 3] - queries have the form (t, x, y); where `t` - is
      in [0, time]; x is in [0, width] and y is in [0, height] * 'target_points'
      - a float32 tensor of shape [batch, num_queries, time, 2] - target points
      of the form (y, x), same ranges as query points * 'visible' - a float32
      tensor of shape [batch, num_queries, time, 1] - visibility flags (1. is
      visible and 0. is not visible)
    loss_weight (float): weight of the loss (default: 1.0)

  Returns:
    loss (float): the total loss
  """
  pred_tracks, track_logits, visible_logits, _ = model(
      video=batch['video'], query_points=batch['query_points']
  )
  huber_loss = huber_coordinate_loss(
      pred_tracks,
      batch['target_points'].transpose(1, 2).flip(-1),
      batch['visible'].transpose(1, 2),
  )
  softmax_loss = coordinate_softmax(
      track_logits,
      batch['target_points'].transpose(1, 2).flip(-1),
      batch['visible'].transpose(1, 2),
  )
  coordinate_loss = 0.1 * huber_loss + 1.0 * softmax_loss
  visibility_loss = F.binary_cross_entropy_with_logits(
      visible_logits, batch['visible'].transpose(1, 2)
  )
  loss = coordinate_loss + visibility_loss
  (loss * loss_weight).backward()
  return loss.item()

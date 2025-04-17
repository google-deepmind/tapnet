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

import dataclasses

import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.losses import base
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import optax


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Huber(base.Loss):
  """Huber loss for point track prediction."""

  delta: float = 1.0

  pred_points: kontext.Key = kontext.REQUIRED
  target_points: kontext.Key = kontext.REQUIRED
  normalize_by: str = "values"

  @typechecked
  def get_values(
      self,
      pred_points: Float["*a 2"],
      target_points: Float["*a 2"],
  ) -> Float["*a 1"]:
    pred_points = jnp.astype(pred_points, jnp.float32)
    target_points = jnp.astype(target_points, jnp.float32)
    target_points = jnp.clip(target_points, 0, 255)
    error = pred_points - target_points
    error = jnp.clip(error, -1e8, 1e8)  # add magnitude bound to prevent nan
    distsqr = jnp.sum(jnp.square(error), axis=-1, keepdims=True)
    dist = jnp.sqrt(distsqr + 1e-12)  # add eps to prevent nan
    loss = jnp.where(
        dist < self.delta,
        distsqr / 2,
        self.delta * (jnp.abs(dist) - self.delta / 2),
    )
    return loss


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class MaskedL1(base.Loss):
  """Masked L1 loss for predicting random image patches."""
  pred_patches: kontext.Key = kontext.REQUIRED
  target_patches: kontext.Key = kontext.REQUIRED
  temporal_mask: kontext.Key = kontext.REQUIRED
  normalize_by: str = "values"
  image_norm: str = "sum"  # "sum" or "mean"

  @typechecked
  def get_values(
      self,
      pred_patches: Float["*B T h w C"],
      target_patches: Float["*B T h w C"],
      temporal_mask: Bool["*B T"],
  ) -> Float["*a 1"]:
    pred_patches = jnp.astype(pred_patches, jnp.float32)
    target_patches = jnp.astype(target_patches, jnp.float32)
    loss = jnp.abs(pred_patches - target_patches)  # * temporal_mask
    if self.image_norm == "sum":
      loss = jnp.sum(loss, axis=[-1, -2, -3]) / 1024.
    elif self.image_norm == "mean":
      loss = jnp.mean(loss, axis=[-1, -2, -3])
    loss = jnp.mean(loss, axis=-1)
    return loss[..., None]


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class CoordinateSoftmaxCrossEntropyWithIntLabels(base.Loss):
  """Softmax cross-entropy loss with integer labels."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"
  pixel_size: int = 256

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Float["*a 2"]
  ) -> Float["*a 1"]:
    logits = jnp.astype(logits, jnp.float32)
    labels = jnp.astype(labels, jnp.float32)
    labels -= 0.5
    labels = jnp.clip(labels, 0, self.pixel_size - 1)
    labels = jnp.round(labels).astype(jnp.int32)
    logits_x, logits_y = jnp.split(logits, 2, axis=-1)
    labels_x, labels_y = jnp.split(labels, 2, axis=-1)
    loss_x = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_x, labels=labels_x.squeeze(-1)
    )[..., None]
    loss_y = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_y, labels=labels_y.squeeze(-1)
    )[..., None]
    return loss_x + loss_y


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Certainty(base.Loss):
  """Loss for point track uncertainty prediction.

  A point prediction is certain if it falls within threshold of ground truth.
  The 3rd term of the loss in Equation (1) of TAPIR paper
  https://arxiv.org/abs/2306.08637
  """

  threshold: float = 1.0

  logits: kontext.Key = kontext.REQUIRED
  pred_points: kontext.Key = kontext.REQUIRED
  target_points: kontext.Key = kontext.REQUIRED
  normalize_by: str = "values"

  @typechecked
  def get_values(
      self,
      logits: Float["*a 1"],
      pred_points: Float["*a 2"],
      target_points: Float["*a 2"],
  ) -> Float["*a 1"]:
    logits = jnp.astype(logits, jnp.float32)
    pred_points = jnp.astype(pred_points, jnp.float32)
    target_points = jnp.astype(target_points, jnp.float32)
    pred_points = jax.lax.stop_gradient(pred_points)
    error = pred_points - target_points
    distsqr = jnp.sum(jnp.square(error), axis=-1, keepdims=True)
    is_certain = (distsqr <= self.threshold**2).astype(logits.dtype)
    loss = optax.sigmoid_binary_cross_entropy(logits, is_certain)
    return loss

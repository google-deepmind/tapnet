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

"""Logging and other experiment utilities."""

import os
from typing import Mapping, Optional

import jax
from jaxline import utils
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf

from tapnet import optimizers


def get_lr_schedule(
    total_steps: int,
    optimizer_config: config_dict.ConfigDict,
) -> optax.Schedule:
  """Build the LR schedule function."""
  base_lr = optimizer_config.base_lr

  schedule_type = optimizer_config.schedule_type
  if schedule_type == 'cosine':
    warmup_steps = (optimizer_config.cosine_decay_kwargs.warmup_steps)
    # Batch scale the other lr values as well:
    init_value = optimizer_config.cosine_decay_kwargs.init_value
    end_value = optimizer_config.cosine_decay_kwargs.end_value

    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end_value)
  elif schedule_type == 'constant_cosine':
    # Convert end_value to alpha, used by cosine_decay_schedule.
    alpha = optimizer_config.constant_cosine_decay_kwargs.end_value / base_lr

    # Number of steps spent in constant phase.
    constant_steps = int(
        optimizer_config.constant_cosine_decay_kwargs.constant_fraction *
        total_steps)
    decay_steps = total_steps - constant_steps

    constant_phase = optax.constant_schedule(value=base_lr)
    decay_phase = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=decay_steps, alpha=alpha)
    schedule_fn = optax.join_schedules(
        schedules=[constant_phase, decay_phase], boundaries=[constant_steps])
  else:
    raise ValueError(f'Unknown learning rate schedule: {schedule_type}')

  return schedule_fn


def make_optimizer(
    optimizer_config: config_dict.ConfigDict,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
  """Construct the optax optimizer with given LR schedule."""
  # Decay learned position embeddings by default.
  weight_decay_exclude_names = ['b']

  optax_chain = []
  if optimizer_config.max_norm > 0:
    optax_chain.append(optax.clip_by_global_norm(optimizer_config.max_norm))

  if optimizer_config.optimizer == 'sgd':
    optax_chain.extend([
        optax.trace(**optimizer_config.sgd_kwargs),
        optimizers.add_weight_decay(
            optimizer_config.weight_decay,
            exclude_names=weight_decay_exclude_names)
    ])
  elif optimizer_config.optimizer == 'adam':
    optax_chain.extend([
        optax.scale_by_adam(**optimizer_config.adam_kwargs),
        optimizers.add_weight_decay(
            optimizer_config.weight_decay,
            exclude_names=weight_decay_exclude_names)
    ])
  else:
    raise ValueError(f'Undefined optimizer {optimizer_config.optimizer}')
  optax_chain.extend([
      optax.scale_by_schedule(lr_schedule),
      optax.scale(-1),
  ])

  optimizer = optax.chain(*optax_chain)
  optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=5)
  return optimizer


class NumpyFileCheckpointer(utils.Checkpointer):
  """A Jaxline checkpointer which saves to numpy files on disk."""

  def __init__(self, config: config_dict.ConfigDict, mode: str):
    self._checkpoint_file = os.path.join(config.checkpoint_dir,
                                         'checkpoint.npy')
    self._checkpoint_state = config_dict.ConfigDict()
    del mode

  def get_experiment_state(self, ckpt_series: str) -> config_dict.ConfigDict:
    """Returns the experiment state for a given checkpoint series."""
    if ckpt_series != 'latest':
      raise ValueError('multiple checkpoint series are not supported')
    return self._checkpoint_state

  def save(self, ckpt_series: str) -> None:
    """Saves the checkpoint."""
    if ckpt_series != 'latest':
      raise ValueError('multiple checkpoint series are not supported')
    exp_mod = self._checkpoint_state.experiment_module
    global_step = self._checkpoint_state.global_step
    f_np = lambda x: np.array(jax.device_get(utils.get_first(x)))
    to_save = {}
    for attr, name in exp_mod.CHECKPOINT_ATTRS.items():
      if name == 'global_step':
        raise ValueError(
            'global_step attribute would overwrite jaxline global step')
      np_params = jax.tree_map(f_np, getattr(exp_mod, attr))
      to_save[name] = np_params
    to_save['global_step'] = global_step

    with tf.io.gfile.GFile(self._checkpoint_file + '_tmp', 'wb') as fp:
      np.save(fp, to_save)
    tf.io.gfile.rename(
        self._checkpoint_file + '_tmp',
        self._checkpoint_file,
        overwrite=True,
    )

  def can_be_restored(self, ckpt_series: str) -> bool:
    """Returns whether or not a given checkpoint series can be restored."""
    if ckpt_series != 'latest':
      raise ValueError('multiple checkpoint series are not supported')
    return tf.io.gfile.exists(self._checkpoint_file)

  def restore(self, ckpt_series: str) -> None:
    """Restores the checkpoint."""
    experiment_state = self.get_experiment_state(ckpt_series)
    with tf.io.gfile.GFile(self._checkpoint_file, 'rb') as fp:
      ckpt_state = np.load(fp, allow_pickle=True).item()
    experiment_state.global_step = int(ckpt_state['global_step'])
    exp_mod = experiment_state.experiment_module
    for attr, name in exp_mod.CHECKPOINT_ATTRS.items():
      setattr(exp_mod, attr, utils.bcast_local_devices(ckpt_state[name]))

  def restore_path(self, ckpt_series: str) -> Optional[str]:
    """Returns the restore path for the checkpoint, or None."""
    if not self.can_be_restored(ckpt_series):
      return None
    return self._checkpoint_file

  def wait_for_checkpointing_to_finish(self) -> None:
    """Waits for any async checkpointing to complete."""

  @classmethod
  def create(
      cls,
      config: config_dict.ConfigDict,
      mode: str,
  ) -> utils.Checkpointer:
    return cls(config, mode)


def default_color_augmentation_fn(
    inputs: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Standard color augmentation for videos.

  Args:
    inputs: A DatasetElement containing the item 'video' which will have
      augmentations applied to it.

  Returns:
    A DatasetElement with all the same data as the original, except that
      the video has augmentations applied.
  """
  zero_centering_image = True
  prob_color_augment = 0.8
  prob_color_drop = 0.2

  frames = inputs['video']
  if frames.dtype != tf.float32:
    raise ValueError('`frames` should be in float32.')

  def color_augment(video: tf.Tensor) -> tf.Tensor:
    """Do standard color augmentations."""
    # Note the same augmentation will be applied to all frames of the video.
    if zero_centering_image:
      video = 0.5 * (video + 1.0)
    video = tf.image.random_brightness(video, max_delta=32. / 255.)
    video = tf.image.random_saturation(video, lower=0.6, upper=1.4)
    video = tf.image.random_contrast(video, lower=0.6, upper=1.4)
    video = tf.image.random_hue(video, max_delta=0.2)
    video = tf.clip_by_value(video, 0.0, 1.0)
    if zero_centering_image:
      video = 2 * (video-0.5)
    return video

  def color_drop(video: tf.Tensor) -> tf.Tensor:
    video = tf.image.rgb_to_grayscale(video)
    video = tf.tile(video, [1, 1, 1, 3])
    return video

  # Eventually applies color augmentation.
  coin_toss_color_augment = tf.random.uniform(
      [], minval=0, maxval=1, dtype=tf.float32)
  frames = tf.cond(
      pred=tf.less(coin_toss_color_augment,
                   tf.cast(prob_color_augment, tf.float32)),
      true_fn=lambda: color_augment(frames),
      false_fn=lambda: frames)

  # Eventually applies color drop.
  coin_toss_color_drop = tf.random.uniform(
      [], minval=0, maxval=1, dtype=tf.float32)
  frames = tf.cond(
      pred=tf.less(coin_toss_color_drop, tf.cast(prob_color_drop, tf.float32)),
      true_fn=lambda: color_drop(frames),
      false_fn=lambda: frames)
  result = {**inputs}
  result['video'] = frames

  return result


def add_default_data_augmentation(ds: tf.data.Dataset) -> tf.data.Dataset:
  return ds.map(
      default_color_augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)

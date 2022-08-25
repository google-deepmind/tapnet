# Copyright 2022 DeepMind Technologies Limited
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

from ml_collections import config_dict

import optax

from tapnet import optimizers


def human_readable_size(size: float) -> str:
  if size >= 1e9:
    return '%.2fB' % (size / 1e9)
  elif size >= 1e6:
    return '%.2fM' % (size / 1e6)
  elif size >= 1e3:
    return '%.2fK' % (size / 1e3)
  else:
    return '%s' % size


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

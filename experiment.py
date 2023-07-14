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

"""A Jaxline script for training and evaluating TAPNet."""

import sys
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jaxline import experiment
from jaxline import platform
from jaxline import utils
from kubric.challenges.point_tracking import dataset
from ml_collections import config_dict

import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from tapnet import supervised_point_prediction
from tapnet import tapir_model
from tapnet import tapnet_model
from tapnet import task
from tapnet.utils import experiment_utils as exputils


class Experiment(experiment.AbstractExperiment):
  """TAPNet experiment.

  This constructs a set of tasks which are used to compute gradients, as well as
  a set of datasets which are passed into every task.  After the tasks compute
  gradients using the data, this experiment will aggregate and apply those
  gradients using an optimizer.

  This experiment is organized around the needs of Haiku's multi transform:
  each task specifies its own forward function, but different tasks need to
  share parameters.  Therefore, the design of this experiment is that each
  task has a forward function, but the global Experiment class is responsible
  for creating the multi transform out of these.  The Experiment class keeps
  track of the variables.  Each task doesn't directly call its own forward
  function.  Instead receives a "wrapped" forward function which contains
  the Haiku state, which injected (via closure) by the Experiment class.

  To do its job, multi transform needs two things: a function
  which initializes all variables, and the set of forward functions that need
  to be called separately, one for each task.  The  latter is just the set of
  forward functions for each task.  The initialization function is
  just a function that calls every forward function sequentially.
  """

  # A map from object properties that will be checkpointed to their name
  # in a checkpoint. Currently we assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(
      self,
      mode: str,
      init_rng: jnp.ndarray,
      config: config_dict.ConfigDict,
  ):
    """Initializes the experiment.

    This includes constructing all of the requested tasks, creating the Haiku
    transforms, and pmapping the update function that will be used throughout
    training.  Currently the two supported training tasks are 'selfsup_tracks'
    and 'kubric'.

    Args:
      mode: either 'train' (for training) or one of the recognized eval modes
        (see Experiment.evaluate).
      init_rng: jax.random.PRNGKey for random number generation.
      config: config options.  See configs/tapnet_config.py for an example.
    """

    super().__init__(mode=mode, init_rng=init_rng)

    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._opt_state = None

    self._optimizer = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    self.point_prediction = (
        supervised_point_prediction.SupervisedPointPrediction(
            config, **config.supervised_point_prediction_kwargs
        )
    )

    def forward(*args, is_training=True, **kwargs):
      shared_modules = self._construct_shared_modules()
      return self.point_prediction.forward_fn(
          *args,
          shared_modules=shared_modules,
          is_training=is_training,
          **kwargs,
      )

    self._transform = hk.transform_with_state(forward)

    # NOTE: We "donate" the `params, state, opt_state` arguments which allows
    # JAX (on some backends) to reuse the device memory associated with these
    # inputs to store the outputs of our function (which also start with
    # `params, state, opt_state`).
    self._update_func = jax.pmap(
        self._update_func, axis_name='i', donate_argnums=(0, 1, 2))

  def _construct_shared_modules(self) -> Mapping[str, task.SharedModule]:
    """Constructs the TAPNet module which is used for all tasks.

    More generally, these are Haiku modules that are passed to all tasks so that
    weights are shared across tasks.

    Returns:
      A dict with a single key 'tapnet_model' containing the tapnet model.
    """
    shared_module_constructors = {
        'tapnet_model': tapnet_model.TAPNet,
        'tapir_model': tapir_model.TAPIR,
    }
    shared_modules = {}

    for shared_mod_name in self.config.shared_modules.shared_module_names:
      ctor = shared_module_constructors[shared_mod_name]
      kwargs = self.config.shared_modules[shared_mod_name + '_kwargs']
      shared_modules[shared_mod_name] = ctor(**kwargs)
    return shared_modules

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #
  def step(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      global_step: chex.Array,
      rng: chex.PRNGKey,
      *unused_args,
      **unused_kwargs,
  ) -> Dict[str, chex.Array]:
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._params, self._state, self._opt_state, scalars = self._update_func(
        self._params,
        self._state,
        self._opt_state,
        inputs,
        rng,
        global_step,
    )

    scalars = utils.get_first(scalars)

    if ((utils.get_first(global_step) + 1) % self.config.evaluate_every) == 0:
      for mode in self.config.eval_modes:
        eval_scalars = self.evaluate(global_step, rng=rng, mode=mode)
        scalars.update(eval_scalars)

    return scalars

  def _initialize_train(self):
    """Initializes the training parameters using the first training input.

    self._multi_transform.init will call the forward_fn, which allows
    Haiku to identify the params that need to be initialized.
    """
    self._train_input = utils.py_prefetch(self._build_train_input)

    total_steps = self.config.training.n_training_steps
    # Scale by the (negative) learning rate.
    # Check we haven't already restored params
    if self._params is None:

      logging.info('Initializing parameters.')

      inputs = next(self._train_input)

      init_net = jax.pmap(
          lambda *a: self._transform.init(*a, is_training=True),
          axis_name='i',
      )

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state.
      init_rng = utils.bcast_local_devices(self.init_rng)
      self._params, self._state = init_net(init_rng, inputs)

    self._lr_schedule = exputils.get_lr_schedule(
        total_steps,
        self.config.optimizer,
    )

    self._optimizer = exputils.make_optimizer(
        self.config.optimizer,
        self._lr_schedule,
    )

    init_opt = jax.pmap(self._optimizer.init, axis_name='i')

    if self._opt_state is None:
      self._opt_state = init_opt(self._params)

  def _build_train_input(
      self,
  ) -> Iterable[Mapping[str, Mapping[str, np.ndarray]]]:
    """Builds the training input.

    For each dataset specified in the config, this will call the appropriate
    constructor for the dataset and then yield elements from each dataset.
    Currently supported datasets are 'selfsup_kinetics_tracks' and 'kubric'.

    Yields:
      Elements of a joint dataset, where each item is a dict.  The keys are
      the dataset names, and the associated values are items generated by
      those dataset classes.
    """

    # Note: a dataset constructor is a function which takes only kwargs from
    # the experiment config, and returns a dataset.
    #
    # A dataset consists of a generator function (calling next() on it will
    # produce a data value)
    dataset_constructors = {
        'kubric': dataset.create_point_tracking_dataset,
    }
    dataset_generators = {}
    for dset_name in self.config.datasets.dataset_names:
      ds_generator = self.create_dataset_generator(
          dataset_constructors,
          dset_name,
          color_augmentation=True,
      )

      dataset_generators[dset_name] = ds_generator

    while True:
      combined_dset = {}
      for dset_name in self.config.datasets.dataset_names:
        next_data = next(dataset_generators[dset_name])
        combined_dset[dset_name] = next_data
      yield combined_dset

  def create_dataset_generator(
      self,
      dataset_constructors: Mapping[str, Callable[..., tf.data.Dataset]],
      dset_name: str,
      color_augmentation: bool = False,
  ) -> Iterator[Mapping[str, np.ndarray]]:
    # Batch data on available devices.
    # Number of devices is unknown when an interpreter reads a config file.
    # Here we re-define batch dims to guide jax for right batching.
    batch_dims = [
        jax.local_device_count(),
        self.config.datasets[dset_name + '_kwargs'].batch_dims,
    ]
    dset_kwargs = dict(self.config.datasets[dset_name + '_kwargs'])
    dset_kwargs['batch_dims'] = []
    ds = dataset_constructors[dset_name](**dset_kwargs)
    if color_augmentation:
      ds = exputils.add_default_data_augmentation(ds)

    for dim in batch_dims[::-1]:
      ds = ds.batch(dim)
    np_ds = tfds.as_numpy(ds)
    return iter(np_ds)

  def _update_func(
      self,
      params: chex.ArrayTree,
      state: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      global_step: chex.Array,
  ) -> Tuple[chex.Array, chex.Array, chex.Array, Mapping[str, chex.Numeric]]:
    """Applies an update to parameters and returns new state."""

    updates = None
    grads, state, scalars = self.point_prediction.get_gradients(
        params,
        state,
        inputs,
        rng,
        global_step,
        self._transform.apply,
        is_training=True,
    )
    scalars = {**scalars}  # Mutable copy.
    if grads is not None:
      grads = jax.lax.psum(grads, axis_name='i')
      task_updates, opt_state = self._optimizer.update(
          grads,
          opt_state,
          params,
      )

      # We accumulate task_updates across tasks; if this is the first task
      # processed, then updates is None and we just initialize with the
      # updates from the first task.
      if updates is None:
        updates = task_updates
      else:
        updates = jax.tree_map(jnp.add, updates, task_updates)
      scalars['gradient_norm'] = optax.global_norm(grads)

    # Grab the learning rate to log before performing the step.
    learning_rate = self._lr_schedule(global_step)

    # Make the BYOL predictor train faster than everything else.
    def multiply_lr(variable_name, variable_update):
      need_fast_lr = False
      for substr in self.config.fast_variables:
        need_fast_lr = need_fast_lr or any([substr in x for x in variable_name])
      if need_fast_lr:
        logging.info('Boosting learning rate for: %s', '/'.join(variable_name))
        return variable_update * 10.
      return variable_update

    mapping_type = type(updates)

    def map_fn(nested_dict, prefix=None):
      prefix = prefix or []
      result = {}
      for k, v in nested_dict.items():
        if isinstance(v, mapping_type):
          result[k] = map_fn(v, prefix + [k])
        else:
          result[k] = multiply_lr(prefix + [k], v)

      return mapping_type(**result)

    updates = map_fn(updates)

    params = optax.apply_updates(params, updates)

    n_params = 0
    for k in params.keys():  # pytype: disable=attribute-error  # numpy-scalars
      for l in params[k]:
        n_params = n_params + np.prod(params[k][l].shape)  # pytype: disable=attribute-error  # numpy-scalars

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars.update({
        'learning_rate': learning_rate,
        'n_params (M)': float(n_params / 1e6),
    })
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #
  def evaluate(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      global_step: chex.Array,
      rng: chex.PRNGKey,
      mode: Optional[str] = None,
      **unused_args,
  ) -> Dict[str, chex.Array]:
    mode = mode or self.mode
    point_prediction_task = self.point_prediction
    forward_fn = self._transform.apply
    eval_scalars = point_prediction_task.evaluate(
        global_step=global_step,
        params=self._params,
        state=self._state,
        rng=rng,
        wrapped_forward_fn=forward_fn,
        mode=mode,
    )
    return {
        f'eval/{mode}/{key}': value for key, value in eval_scalars.items()
    }


def main(_):
  flags.mark_flag_as_required('config')
  # Keep TF off the GPU; otherwise it hogs all the memory and leaves none for
  # JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')
  platform.main(
      Experiment,
      sys.argv[1:],
      checkpointer_factory=exputils.NumpyFileCheckpointer.create,
  )

if __name__ == '__main__':
  app.run(main)

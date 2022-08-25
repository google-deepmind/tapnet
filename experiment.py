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

"""A Jaxline script for training and evaluating TAPNet."""

import functools
import os
import sys

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp

from jaxline import experiment
from jaxline import platform
from jaxline import utils
from ml_collections import config_dict

import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from tapnet import kubric_task
from kubric.challenges.point_tracking import dataset

from tapnet import tapnet_model
from tapnet.utils import experiment_utils as exputils


FLAGS = flags.FLAGS


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
      mode: either 'train' (for training) or one of the recognized eval
        modes (see Experiment.evaluate).
      init_rng: jax.random.PRNGKey for random number generation.
      config: config options.  See configs/tapnet_config.py for an example.
    """

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)

    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    self._task_constructors = {
        'kubric': kubric_task.KubricTask,
    }

    self._tasks = {}

    for task in config.tasks.task_names:
      kwargs = self.config['tasks'][task + '_kwargs']
      self._tasks[task] = self._task_constructors[task](**kwargs)

    # NOTE: creating the multi transform causes forward_fn() to be called on all
    # tasks in order to initialize all the variables in the model.  The input
    # will be the full training dict.  Tasks that exclusively re-use variables
    # from other tasks (e.g. eval-only tasks) can simply return without doing
    # anything.
    self._multi_transform = hk.multi_transform_with_state(
        self._forward_fns_for_multi_transform)

    # NOTE: We "donate" the `params, state, opt_state` arguments which allows
    # JAX (on some backends) to reuse the device memory associated with these
    # inputs to store the outputs of our function (which also start with
    # `params, state, opt_state`).
    self._update_func = jax.pmap(
        self._update_func, axis_name='i', donate_argnums=(0, 1, 2))

  def _construct_shared_modules(self):
    """Constructs the TAPNet module which is used for all tasks.

    More generally, these are Haiku modules that are passed to all tasks so that
    weights are shared across tasks.

    Returns:
      A dict with a single key 'tapnet_model' containing the tapnet model.
    """
    shared_module_constructors = {
        'tapnet_model': tapnet_model.TAPNet,
    }
    shared_modules = {}

    for shared_mod_name in self.config.shared_modules.shared_module_names:
      ctor = shared_module_constructors[shared_mod_name]
      kwargs = self.config.shared_modules[shared_mod_name + '_kwargs']
      shared_modules[shared_mod_name] = ctor(**kwargs)
    return shared_modules

  def _forward_fns_for_multi_transform(self):
    """Creates a function that can be called to initialize the network.

    Given an input dict, the returned function will call forward on all of the
    tasks, which allows Haiku to discover the required parameters.  The set of
    modules (created via _construct_shared_modules) will be passed to all tasks
    to enable weight sharing across tasks.

    These are the functions that are required for haiku.multi_transform; the
    joint_forward_fn will initialize all variables, and we'll also return
    a set of forward_fn's that will allow each task to call its own forward
    function and have haiku parameters injected.

    Returns:
      joint_forward_fn: a function that will call forward on all tasks.
      forward_fns: a dict of separate forward functions that can be called
        for every task, keyed by task name.
    """
    shared_modules = self._construct_shared_modules()

    def joint_forward_fn(inputs, is_training=True):
      for task_name in self.config.tasks.task_names:
        self._tasks[task_name].forward_fn(
            inputs, is_training=is_training, shared_modules=shared_modules)

    forward_fns = dict()
    for task_name in self.config.tasks.task_names:
      forward_fn = self._tasks[task_name].forward_fn
      forward_fns[task_name] = functools.partial(
          forward_fn, shared_modules=shared_modules)
    return joint_forward_fn, forward_fns

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #
  def step(
      self,
      global_step: int,
      rng: jnp.ndarray,
      *unused_args,
      **unused_kwargs,
  ) -> dict[str, np.ndarray]:
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._params, self._state, self._opt_state, scalars = (
        self._update_func(
            self._params,
            self._state,
            self._opt_state,
            inputs,
            rng,
            global_step,
        ))

    scalars = utils.get_first(scalars)

    # Save final checkpoint.
    if self.config.save_final_checkpoint_as_npy:
      global_step_value = utils.get_first(global_step)
      if global_step_value == FLAGS.config.get('training_steps', 1) - 1:
        f_np = lambda x: np.array(jax.device_get(utils.get_first(x)))
        np_params = jax.tree_map(f_np, self._params)
        np_state = jax.tree_map(f_np, self._state)
        path_npy = os.path.join(FLAGS.config.checkpoint_dir, 'checkpoint.npy')
        with tf.io.gfile.GFile(path_npy, 'wb') as fp:
          np.save(fp, (np_params, np_state))
        logging.info('Saved final checkpoint at %s', path_npy)

    task_for_eval_mode = {
        'eval_kubric': 'kubric',
        'eval_jhmdb': 'kubric',
        'eval_robotics_points': 'kubric',
    }
    for mode in FLAGS.config.eval_modes:
      if (global_step % FLAGS.config.evaluate_every) != 0:
        break

      logging.info('Launch %s at step %d', mode, global_step)
      task_name = task_for_eval_mode[mode]
      task = self._tasks[task_name]
      forward_fn = self._multi_transform.apply[task_name]
      eval_scalars = task.evaluate(
          global_step=global_step,
          params=self._params,
          state=self._state,
          rng=rng,
          wrapped_forward_fn=forward_fn,
          mode=mode,
      )
      eval_scalars = {
          f'eval/{mode}/{key}': value for key, value in eval_scalars.items()
      }
      scalars.update(eval_scalars)

    return scalars

  def _initialize_train(self):
    """Initializes the training parameters using the first training input.

    self._multi_transform.init will call the joint_forward_fn, which calls
    the forward function for every task and allows Haiku to identify the
    params that need to be initialized.
    """
    self._train_input = utils.py_prefetch(self._build_train_input)

    total_steps = self.config.training.n_training_steps
    # Scale by the (negative) learning rate.
    # Check we haven't already restored params
    if self._params is None:
      logging.info(
          'Initializing parameters rather than restoring from checkpoint.',)

      inputs = next(self._train_input)

      init_net = jax.pmap(
          lambda *a: self._multi_transform.init(*a, is_training=True),
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
      self._opt_state = {
          task_name: init_opt(self._params) for task_name in self._tasks
      }

  def _build_train_input(self):
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
      )

      dataset_generators[dset_name] = ds_generator()

    while True:
      combined_dset = {}
      for dset_name in self.config.datasets.dataset_names:
        next_data = next(dataset_generators[dset_name])
        combined_dset[dset_name] = next_data
      yield combined_dset

  def create_dataset_generator(self, dataset_constructors, dset_name):
    # Batch data on available devices.
    # Number of devices is unknown when an interpreter reads a config file.
    # Here we re-define batch dims to guide jax for right batching.
    batch_dims = [
        jax.local_device_count(),
        self.config.datasets[dset_name + '_kwargs'].batch_dims,
    ]
    dset_kwargs = dict(self.config.datasets[dset_name + '_kwargs'])
    dset_kwargs['batch_dims'] = batch_dims
    ds = dataset_constructors[dset_name](**dset_kwargs)
    np_ds = tfds.as_numpy(ds)

    def ds_generator(data_set=np_ds):
      yield from data_set

    return ds_generator

  def _update_func(
      self,
      params,
      state,
      opt_state,
      inputs,
      rng,
      global_step,
  ):
    """Applies an update to parameters and returns new state."""

    updates = None
    all_scalars = {}
    for task_name in self.config.tasks.task_names:
      grads, state, scalars = self._tasks[task_name].get_gradients(
          params,
          state,
          inputs,
          rng,
          global_step,
          self._multi_transform.apply[task_name],
          is_training=True,
      )
      if grads is not None:
        grads = jax.lax.psum(grads, axis_name='i')
        task_updates, opt_state[task_name] = self._optimizer.update(
            grads,
            opt_state[task_name],
            params,
        )

        # We accumulate task_updates across tasks; if this is the first task
        # processed, then updates is None and we just initialize with the
        # updates from the first task.
        if updates is None:
          updates = task_updates
        else:
          updates = jax.tree_map(jnp.add, updates, task_updates)
        all_scalars[task_name + '_gradient_norm'] = optax.global_norm(grads)
      all_scalars.update({task_name + '/' + k: v for k, v in scalars.items()})

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
    for k in params.keys():
      for l in params[k]:
        n_params = n_params + np.prod(params[k][l].shape)

    # Scalars to log (note: we log the mean across all hosts/devices).
    all_scalars.update({
        'learning_rate': learning_rate,
        'n_params (M)': float(n_params / 1e6),
    })
    all_scalars = jax.lax.pmean(all_scalars, axis_name='i')

    return params, state, opt_state, all_scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #
  def evaluate(self, global_step: jnp.ndarray, rng: jnp.ndarray, **unused_args):
    del global_step, rng, unused_args


def main(_):
  flags.mark_flag_as_required('config')
  platform.main(Experiment, sys.argv[1:])

if __name__ == '__main__':
  app.run(main)

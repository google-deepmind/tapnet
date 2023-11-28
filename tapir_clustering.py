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

"""Clustering TAPIR tracks based on independent motion."""

import functools
import time
from typing import NamedTuple

from einshape import jax_einshape as einshape
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tapnet import tapir_model


class TrainingState(NamedTuple):
  """Container for the training state."""

  params: hk.Params
  state: hk.State
  opt_state: optax.OptState
  rng: jax.Array
  step: jax.Array


def make_projection_matrix(pred_mat, fourdof=True):
  """Convert predicted projection matrix parameters to a projection matrix."""
  pred_mat = einshape('n(coi)->ncoi', pred_mat, o=3, i=4)

  # This runs Gram-Schmidt to create an orthonormal matrix from the input 3x3
  # matrix that comes from a neural net.
  #
  # We run gradient clipping on the backward pass because the matrix might be
  # badly conditioned.
  @jax.custom_vjp
  def f(x):
    return x

  def f_fwd(x):
    return f(x), tuple()

  def f_bwd(_, g):
    return (jnp.clip(g, -100, 100),)

  f.defvjp(f_fwd, f_bwd)
  pred_mat = f(pred_mat)
  if fourdof:
    orth1 = jnp.ones_like(pred_mat[..., 0:1, :-1]) * jnp.array([0.0, 0.0, 1.0])
    orth2 = pred_mat[..., 1:2, :-1] * jnp.array([1.0, 1.0, 0.0])
  else:
    orth1 = pred_mat[..., 0:1,:-1]
    orth1 = orth1 / jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.square(orth1), axis=-1, keepdims=True), 1e-12)
    )
    orth2 = pred_mat[..., 1:2, :-1]
    orth2 = orth2 - orth1 * jnp.sum(orth2 * orth1, axis=-1, keepdims=True)
  orth2 = orth2 / jnp.sqrt(
      jnp.maximum(jnp.sum(jnp.square(orth2), axis=-1, keepdims=True), 1e-12)
  )
  orth3 = pred_mat[..., 2:3, :-1]
  if fourdof:
    orth3 *= jnp.array([1.0, 1.0, 0.0])
  else:
    orth3 = orth3 - orth1 * jnp.sum(orth3 * orth1, axis=-1, keepdims=True)
  orth3 = orth3 - orth2 * jnp.sum(orth3 * orth2, axis=-1, keepdims=True)
  orth3 = orth3 / jnp.sqrt(
      jnp.maximum(jnp.sum(jnp.square(orth3), axis=-1, keepdims=True), 1e-12)
  )

  cross_prod = jnp.cross(orth1, orth2)
  orth3 = orth3 * jnp.sign(jnp.sum(cross_prod * orth3, axis=-1, keepdims=True))

  orth = jnp.concatenate([orth3, orth2, orth1], axis=-2)
  pred_mat = jnp.concatenate([orth, pred_mat[..., -1:]], axis=-1)
  return pred_mat


def project(pred_mat, pos_pred, cam_focal_length):
  """Project 3D points to 2D, with penalties for depth out-of-range."""
  pos_pred = jnp.concatenate(
      [pos_pred[..., :3], pos_pred[..., 0:1] * 0 + 1], axis=-1
  )
  pred_pos = jnp.einsum('fcoi,nci->nfco', pred_mat, pos_pred)
  depth = jnp.minimum(2.0, jnp.maximum(pred_pos[..., 2:3] + 1.0, 0.5))
  oob = jnp.maximum(pred_pos[..., 2:3] - 2.0, 0.0) + jnp.maximum(
      0.5 - pred_pos[..., 2:3], 0.0
  )
  all_pred = pred_pos[..., 0:2] * cam_focal_length / depth
  all_pred = (
      all_pred
      + 0.1 * jax.random.normal(hk.next_rng_key(), shape=oob.shape) * oob
  )
  return all_pred, depth[..., 0]


def forward(
    fr_idx,
    pts_idx,
    pts,
    vis,
    num_cats=4,
    is_training=True,
    sequence_boundaries=tuple(),
    fourdof=True,
    cam_focal_length=1.0,
):
  """Model forward pass."""

  def bn(x):
    return hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(
        x, is_training=is_training
    )

  pts_shape = pts.shape
  pts = jnp.reshape(pts * vis[..., jnp.newaxis], [pts.shape[0], -1])
  pt_state = hk.get_parameter('point_state', [pts_shape[0], 64], init=jnp.zeros)

  def centroid_init(shp, dtype):
    del shp  # unused
    centroid_weights = jax.nn.one_hot(
        jax.random.randint(hk.next_rng_key(), [384], 0, pts.shape[0]),
        pts.shape[0],
        axis=0,
    )
    centroids = jnp.transpose(centroid_weights) @ pts
    centroid_vis = jnp.transpose(centroid_weights) @ vis
    time_weight = centroid_vis
    centroids = jnp.concatenate([centroids, time_weight * 100.0], axis=1)
    centroids = jnp.transpose(centroids)
    return jnp.array(centroids, dtype=dtype)

  centroids = hk.get_parameter(
      'centroids', [pts_shape[1] * 3, 384], init=centroid_init
  )
  time_weight = jnp.abs(centroids[pts_shape[1] * 2 :, :]) / 100.0
  centroids = centroids[: pts_shape[1] * 2, :]
  vis_tile = jnp.reshape(
      jnp.tile(vis[:, :, jnp.newaxis], [1, 1, 2]), [pts.shape[0], -1]
  )
  tw_tile = jnp.reshape(
      jnp.tile(time_weight[:, jnp.newaxis, :], [1, 2, 1]), [-1, 384]
  )

  dists = jnp.square(pts * vis_tile) @ jnp.square(tw_tile)
  dists -= 2 * (pts * vis_tile) @ (centroids * tw_tile)
  dists += jnp.square(vis_tile) @ jnp.square(centroids * tw_tile)
  dists = jnp.exp(-dists * 10.0)

  dists = dists / jnp.maximum(jnp.sum(dists, axis=-1, keepdims=True), 1e-8)
  pt_state += hk.Linear(64)(dists)
  frame_state_nosmooth = hk.get_parameter(
      'frame_state',
      [pts_shape[1], 64],
      init=hk.initializers.TruncatedNormal(1.0),
  )
  conv = hk.Conv1D(64, 128, feature_group_count=64)
  frame_state = []
  for bnd in sequence_boundaries:
    frame_state.append(conv(frame_state_nosmooth[bnd[0] : bnd[1]]))
  frame_state = jnp.concatenate(frame_state, axis=0)

  frame_state = bn(frame_state)
  pt_state = bn(pt_state)
  state = jax.nn.relu(hk.Linear(64)(pt_state))
  state += hk.Linear(64)(jax.nn.relu(bn(hk.Linear(32)(state))))
  state += hk.Linear(64)(jax.nn.relu(bn(hk.Linear(32)(state))))

  truncated_normal = hk.initializers.TruncatedNormal

  base_pred = hk.get_parameter(
      'cat_pred_base',
      [3 * 64 * pts_shape[0], num_cats],
      init=truncated_normal(1.0),
  )
  fork1_pred = hk.get_parameter(
      'cat_pred_fork1',
      [3 * 64 * pts_shape[0], num_cats],
      init=lambda *args: truncated_normal(1.0)(*args) * 0.0001 + base_pred,
  )
  fork2_pred = hk.get_parameter(
      'cat_pred_fork2',
      [3 * 64 * pts_shape[0], num_cats],
      init=lambda *args: truncated_normal(1.0)(*args) * 0.0001 + base_pred,
  )

  def mul(mat):
    mat = einshape('(pio)c->pcio', mat, i=64, o=3)
    return jnp.einsum('pcio,pi->pco', mat, state) * 0.01

  pos_pred_base = mul(base_pred)[pts_idx]
  pos_pred_fork1 = mul(fork1_pred)[pts_idx]
  pos_pred_fork2 = mul(fork2_pred)[pts_idx]

  state = frame_state
  state = jax.nn.relu(hk.Linear(128)(state))
  state += hk.Linear(128)(bn(jax.nn.relu(hk.Linear(64)(state))))
  state += hk.Linear(128)(bn(jax.nn.relu(hk.Linear(64)(state))))
  state = state * 0.01

  base = hk.get_parameter(
      'mat_pred_base',
      [state.shape[-1], num_cats * 12],
      init=hk.initializers.TruncatedNormal(1.0),
  )
  fork1 = hk.get_parameter(
      'mat_pred_fork1',
      [state.shape[-1], num_cats * 12],
      init=hk.initializers.TruncatedNormal(1.0),
  )
  fork2 = hk.get_parameter(
      'mat_pred_fork2',
      [state.shape[-1], num_cats * 12],
      init=hk.initializers.TruncatedNormal(1.0),
  )
  pred_mat_base = state @ base
  pred_mat_fork1 = state @ fork1
  pred_mat_fork2 = state @ fork2

  pred_mat_base = make_projection_matrix(pred_mat_base, fourdof)[fr_idx]
  pred_mat_fork1 = make_projection_matrix(pred_mat_fork1, fourdof)[fr_idx]
  pred_mat_fork2 = make_projection_matrix(pred_mat_fork2, fourdof)[fr_idx]

  if not is_training:
    pred_pos_all, depth_all = project(
        pred_mat_base, pos_pred_base, cam_focal_length
    )
    return pred_pos_all, depth_all
  else:
    return {
        'pos_pred_base': pos_pred_base,
        'pos_pred_fork1': pos_pred_fork1,
        'pos_pred_fork2': pos_pred_fork2,
        'pred_mat_base': pred_mat_base,
        'pred_mat_fork1': pred_mat_fork1,
        'pred_mat_fork2': pred_mat_fork2,
    }


# Create the loss.
@hk.transform_with_state
def loss_fn(
    data,
    num_cats=4,
    delete_mode=False,
    sequence_boundaries=tuple(),
    final_num_cats=28,
    use_em=False,
    fourdof=True,
    cam_focal_length=1.0,
):
  """Computes the (scalar) LM loss on `data` w.r.t. params."""
  pts, vis, _ = data
  pts_idx = jax.random.permutation(hk.next_rng_key(), pts.shape[0])[:2048]
  fr_idx = jax.random.permutation(hk.next_rng_key(), pts.shape[1])[:1024]

  fwd = forward(
      fr_idx,
      pts_idx,
      pts,
      vis,
      num_cats=num_cats,
      sequence_boundaries=sequence_boundaries,
      fourdof=fourdof,
      cam_focal_length=cam_focal_length,
  )

  pts = pts[pts_idx][:, fr_idx]
  vis = vis[pts_idx][:, fr_idx]

  def do_fork(base, fork1, fork2, i, chunk=1):
    chunk1 = base[..., : i * chunk]
    chunk2 = fork1[..., i * chunk : ((1 + i) * chunk)]
    chunk3 = fork2[..., i * chunk : ((1 + i) * chunk)]
    chunk4 = base[..., (1 + i) * chunk :]
    return jnp.concatenate([chunk1, chunk2, chunk3, chunk4], axis=-1)

  def do_delete(base, i, chunk=1):
    chunk1 = base[..., : i * chunk]
    chunk4 = base[..., (1 + i) * chunk :]
    return jnp.concatenate([chunk1, chunk4], axis=-1)

  losses = []
  sum_vis = jnp.sum(vis)

  # The following is the recursive cluster splitting and deleting algorithm:
  # for every cluster, we 'split' it, creating 2 new clusters, or delete it.
  # We optimize for every candidate cluster to split/delete, and choose
  # the split/delete that minimizes the overall error.
  if delete_mode:
    all_pred, _ = project(
        fwd['pred_mat_base'], fwd['pos_pred_base'], cam_focal_length
    )
    all_err = get_err(pts, vis, all_pred)
    for i in range(fwd['pred_mat_base'].shape[-3]):
      err_i = do_delete(all_err, i)
      losses.append(loss_internal(err_i, sum_vis, use_em=use_em))
  else:
    all_pred_base, _ = project(
        fwd['pred_mat_base'], fwd['pos_pred_base'], cam_focal_length
    )
    all_err_base = get_err(pts, vis, all_pred_base)
    all_pred_fork1, _ = project(
        fwd['pred_mat_fork1'], fwd['pos_pred_fork1'], cam_focal_length
    )
    all_err_fork1 = get_err(pts, vis, all_pred_fork1)
    all_pred_fork2, _ = project(
        fwd['pred_mat_fork2'], fwd['pos_pred_fork2'], cam_focal_length
    )
    all_err_fork2 = get_err(pts, vis, all_pred_fork2)
    for i in range(fwd['pred_mat_base'].shape[-3]):
      err_i = do_fork(all_err_base, all_err_fork1, all_err_fork2, i)
      losses.append(loss_internal(err_i, sum_vis, use_em=use_em))
  if delete_mode:
    topk, _ = jax.lax.top_k(-jnp.array(losses), num_cats - final_num_cats + 3)
    accum_loss = jnp.mean(-topk)
  else:
    accum_loss = jnp.min(jnp.array(losses))
  return accum_loss, jnp.array(losses)


def huber(x):
  sqrt_x = jnp.sqrt(jnp.maximum(x, 1e-12))
  return jnp.where(x < 0.004, x, 0.004 * (2 * sqrt_x - 0.004)) * 100.0


def get_err(pts, vis, all_pred):
  tmp = pts[:, :, jnp.newaxis, :] - all_pred
  tmp = jnp.sum(jnp.square(tmp) * vis[:, :, jnp.newaxis, jnp.newaxis], axis=-1)

  return jnp.sum(tmp, axis=1)


def loss_internal(err_summed, sum_vis, use_em, em_variance=0.0001):
  """Computes cluster assignments and loss given per-cluster error."""
  if use_em:
    # In typical EM for gaussian mixture models, you keep the estimates of the
    # prior probabilities for each mixture component (often called pi) across
    # iterations.  We could in principle do it that way for this code, but
    # it's hard to say what we should do with these values for the
    # 'splitting' and 'deleting' steps of the algorithm.  Therefore, it's
    # simpler to just estimate them on-the-fly based on the cluster
    # membership probabilities.  This needs to be done iteratively,
    # but it converges extremely fast to something that's good enough.
    err_normalized = err_summed - jnp.min(err_summed, axis=1, keepdims=True)
    err_exp = jnp.exp(-err_normalized / em_variance)
    wts = jnp.zeros([1, err_exp.shape[1]]) + 1.0 / err_exp.shape[1]
    for _ in range(3):
      wts = err_exp * wts / jnp.sum(err_exp * wts, axis=1, keepdims=True)
      wts = jnp.sum(wts, axis=0, keepdims=True)
      wts = jnp.maximum(wts, 1e-8)
      wts = wts / jnp.sum(wts)

    min_loss = (
        -jnp.sum(
            jax.scipy.special.logsumexp(
                -err_summed / em_variance, b=wts, axis=1
            )
        )
        / sum_vis
        * em_variance
    )

    return min_loss
  else:
    min_loss = jnp.sum(jnp.min(err_summed, axis=1)) / sum_vis

    return min_loss


def loss_fn_wrapper(*args, **kwargs):
  (loss, aux), state = loss_fn.apply(*args, **kwargs)
  return loss, (state, aux)


def update(
    state,
    data,
    lr_mul=1.0,
    num_cats=4,
    delete_mode=False,
    sequence_boundaries=tuple(),
    optimiser=None,
    final_num_cats=28,
    use_em=False,
    fourdof=True,
    cam_focal_length=1.0,
):
  """Does an SGD step and returns metrics."""
  rng, new_rng = jax.random.split(state.rng)
  loss_and_grad_fn = jax.value_and_grad(
      functools.partial(
          loss_fn_wrapper,
          num_cats=num_cats,
          delete_mode=delete_mode,
          sequence_boundaries=sequence_boundaries,
          final_num_cats=final_num_cats,
          use_em=use_em,
          fourdof=fourdof,
          cam_focal_length=cam_focal_length,
      ),
      has_aux=True,
  )
  (loss, (new_state, losses)), gradients = loss_and_grad_fn(
      state.params, state.state, rng, data
  )

  updates, new_opt_state = optimiser.update(gradients, state.opt_state)
  updates = jax.tree_map(lambda x: x * lr_mul, updates)
  new_params = optax.apply_updates(state.params, updates)

  new_state = TrainingState(
      params=new_params,
      state=new_state,
      opt_state=new_opt_state,
      rng=new_rng,
      step=state.step + 1,
  )

  metrics = {
      'step': state.step,
      'loss': loss,
      'losses': losses,
  }
  return new_state, metrics


@hk.transform_with_state
def forward_fn(pts_idx, pts, vis, num_cats=4, sequence_boundaries=tuple()):
  """Test-time forward function."""
  preds_all, depth_all = forward(
      jnp.arange(pts.shape[1], dtype=jnp.int32),
      pts_idx,
      pts,
      vis,
      num_cats=num_cats,
      is_training=False,
      sequence_boundaries=sequence_boundaries,
  )

  pts = pts[pts_idx]
  vis = vis[pts_idx]
  err = jnp.sum(jnp.square(pts[:, :, jnp.newaxis, :] - preds_all), axis=-1)
  return err * vis[:, :, jnp.newaxis], preds_all, depth_all


def pts_eval(
    state,
    pts_idx,
    pts,
    vis,
    num_cats=4,
    sequence_boundaries=tuple(),
):
  """Evaluate the errors for some points."""
  (err, pred_all, depth_all), _ = forward_fn.apply(
      state.params,
      state.state,
      state.rng,
      pts_idx,
      pts,
      vis,
      num_cats=num_cats,
      sequence_boundaries=sequence_boundaries,
  )

  return err, pred_all, depth_all


def init(rng, data, num_cats=1, sequence_boundaries=tuple(), optimiser=None):
  rng, init_rng = jax.random.split(rng)
  initial_params, initial_state = loss_fn.init(
      init_rng, data, num_cats=num_cats, sequence_boundaries=sequence_boundaries
  )
  initial_opt_state = optimiser.init(initial_params)
  return TrainingState(
      params=initial_params,
      state=initial_state,
      opt_state=initial_opt_state,
      rng=rng,
      step=jnp.array(0),
  )


def compute_clusters(
    separation_tracks_dict,
    separation_visibility_dict,
    demo_episode_ids,
    separation_video_shapes,
    query_features=None,
    final_num_cats=15,
    max_num_cats=25,
    low_visibility_threshold=0.1,
    use_em=False,
    fourdof=True,
    cam_focal_length=1.0,
):
  """Compute clustering.

  Args:
    separation_tracks_dict: dict of tracks keyed by episode id, each of shape
      [num_points, num frames, 2].
    separation_visibility_dict: dict of visibility values keyed by episode id,
      each of shape [num_points, num frames].
    demo_episode_ids: demo episode ids
    separation_video_shapes: dict of video sizes (i.e. num_frames, height,
      width, channels), keyed by episode id. Currently assumes that they are all
      the same height, width.
    query_features: query features associated with each points (short ones will
      be removed)
    final_num_cats: the number of output clusters
    max_num_cats: the maximum number of clusters after splitting, before
      beginning to merge.
    low_visibility_threshold: throw out tracks with less than this fraction of
      visible frames.
    use_em: if True, use an EM-style soft cluster assignment.  Not used in
      RoboTAP, but empirically it can prevent the optimization from getting
      stuck in local minima.
    fourdof: if True (default), restrict the 3D transformations between frames
      to be four degrees of freedom (i.e. depth, 2D translation, in-plane
      rotation).  Otherwise allow for full 6-degree-of-freedom transformations
      between frames for objects.  Note that 6DOF is likely to result in
      objects being merged, because the model can use 3D rotation to explain
      different 2D translations.
    cam_focal_length: Camera focal length.  Camera projection matrix is assumed
      to have the form diag([f, f, 1.0]) @ [R,t] where R and t are the learned
      rotation matrix and translation vector and f is camera_focal_length.  The
      optimization is typically not very sensitive to this; we used 1.0 for
      RoboTAP, which is not correct for our cameras.

  Returns:
    A dict, where low-visibility points have been removed.  "classes" is
      the class id's for remaining points, and "sum_error" is the sum of error
      for visible points.
  """
  iters_before_split = 500
  num_iters = (
      max_num_cats + (max_num_cats - final_num_cats) - 1
  ) * iters_before_split

  separation_tracks = np.concatenate(
      [separation_tracks_dict[x] for x in demo_episode_ids], axis=1
  )
  separation_visibility = np.concatenate(
      [separation_visibility_dict[x] for x in demo_episode_ids], axis=1
  )

  enough_visible = (
      np.mean(separation_visibility, axis=-1) > low_visibility_threshold
  )

  separation_tracks = separation_tracks[enough_visible]
  separation_visibility = separation_visibility[enough_visible]
  if query_features is not None:
    query_features = jax.tree_map(
        lambda x: x[:, enough_visible] if len(x.shape) > 1 else x,
        query_features,
    )
  separation_tracks_dict = jax.tree_map(
      lambda x: x[enough_visible], separation_tracks_dict
  )
  separation_visibility_dict = jax.tree_map(
      lambda x: x[enough_visible], separation_visibility_dict
  )

  cur = 0
  sequence_boundaries = []
  for shp in [separation_video_shapes[x] for x in demo_episode_ids]:
    sequence_boundaries.append((cur, cur + shp[0]))
    cur += shp[0]
  sequence_boundaries = tuple(sequence_boundaries)

  # Create the optimiser.
  optimiser = optax.chain(
      optax.clip_by_global_norm(1e-3),
      optax.adam(5e-2, b1=0.9, b2=0.99),
  )

  # Initialise the model parameters.
  rng = jax.random.PRNGKey(42)

  shp = separation_video_shapes[demo_episode_ids[0]]
  data = (
      jnp.array(separation_tracks / np.array([shp[2], shp[1]])),
      jnp.array(separation_visibility),
  )

  state = init(
      rng,
      data + (1,),
      num_cats=1,
      sequence_boundaries=sequence_boundaries,
      optimiser=optimiser,
  )  # +(np.zeros([100],dtype=np.int32),np.zeros([100],dtype=np.int32)))

  # Start training (note we don't include any explicit eval in this example).
  prev_time = time.time()
  log_every = 10
  num_cats = 1
  loss_curve = []
  loss_moving_average = 0
  num_since_fork = 1000
  delete_mode = False
  need_compile = True

  for step in range(num_iters):
    if step % iters_before_split == iters_before_split - 1:
      if delete_mode:
        num_cats -= 1
        to_delete = np.argmin(loss_moving_average)
        print('deleting:' + str(to_delete) + '; new num_cats:' + str(num_cats))

        def do_delete(val, chunk=1):
          val = np.array(val)
          lb = to_delete * chunk  # pylint: disable=cell-var-from-loop
          ub = (to_delete + 1) * chunk  # pylint: disable=cell-var-from-loop
          return np.concatenate([val[:, :lb], val[:, ub:]], axis=1)

        def delete_dict(param_dict):
          param_dict['cat_pred_base'] = do_delete(param_dict['cat_pred_base'])  # pylint: disable=cell-var-from-loop
          param_dict['cat_pred_fork1'] = do_delete(param_dict['cat_pred_fork1'])  # pylint: disable=cell-var-from-loop
          param_dict['cat_pred_fork2'] = do_delete(param_dict['cat_pred_fork2'])  # pylint: disable=cell-var-from-loop
          param_dict['mat_pred_base'] = do_delete(  # pylint: disable=cell-var-from-loop
              param_dict['mat_pred_base'], chunk=12
          )
          param_dict['mat_pred_fork1'] = do_delete(  # pylint: disable=cell-var-from-loop
              param_dict['mat_pred_fork1'], chunk=12
          )
          param_dict['mat_pred_fork2'] = do_delete(  # pylint: disable=cell-var-from-loop
              param_dict['mat_pred_fork2'], chunk=12
          )

        delete_dict(state.params['~'])
        delete_dict(state.opt_state[1][0].mu['~'])
        delete_dict(state.opt_state[1][0].nu['~'])

      else:
        num_cats += 1
        to_split = jnp.argmin(loss_moving_average)
        print('splitting:' + str(to_split) + '; new num_cats:' + str(num_cats))

        def do_fork(base, fork1, fork2, chunk=1, noise=0.0, mul=1.0):
          base = np.array(base)
          fork1 = np.array(fork1)
          fork2 = np.array(fork2)
          lb = to_split * chunk  # pylint: disable=cell-var-from-loop
          ub = (to_split + 1) * chunk  # pylint: disable=cell-var-from-loop
          base[:, lb:ub] = fork1[:, lb:ub]
          base = np.concatenate([base, fork2[:, lb:ub]], axis=-1)

          def reinit(fork):
            fork = np.concatenate(
                [
                    fork,
                    (
                        fork[:, lb:ub]
                        + np.random.normal(size=[fork.shape[0], chunk]) * noise
                    ),
                ],
                axis=-1,
            )
            fork[:, lb:ub] = (
                fork[:, lb:ub]
                + np.random.normal(size=[fork.shape[0], chunk]) * noise
            )
            if noise > 0:
              fork = np.copy(base) + np.random.normal(size=base.shape) * noise
            fork = fork * mul
            return fork

          return base, reinit(fork1), reinit(fork2)

        def fork_dict(param_dict, noise=0.0, mul=1.0):
          new_cpb, new_cpf1, new_cpf2 = do_fork(  # pylint: disable=cell-var-from-loop
              param_dict['cat_pred_base'],
              param_dict['cat_pred_fork1'],
              param_dict['cat_pred_fork2'],
              noise=noise,
              mul=mul,
          )
          param_dict['cat_pred_base'] = new_cpb
          param_dict['cat_pred_fork1'] = new_cpf1
          param_dict['cat_pred_fork2'] = new_cpf2
          new_mpb, new_mpf1, new_mpf2 = do_fork(  # pylint: disable=cell-var-from-loop
              param_dict['mat_pred_base'],
              param_dict['mat_pred_fork1'],
              param_dict['mat_pred_fork2'],
              chunk=12,
              noise=noise,
              mul=mul,
          )
          param_dict['mat_pred_base'] = new_mpb
          param_dict['mat_pred_fork1'] = new_mpf1
          param_dict['mat_pred_fork2'] = new_mpf2

        fork_dict(state.params['~'], noise=0.000001)
        fork_dict(state.opt_state[1][0].mu['~'], mul=0.0)
        fork_dict(state.opt_state[1][0].nu['~'], mul=1.0)

        state = TrainingState(
            params=state.params,
            state=state.state,
            opt_state=optimiser.init(state.params),
            rng=state.rng,
            step=state.step,
        )

        delete_mode = num_cats == max_num_cats
      num_since_fork = 0
      loss_moving_average = 0
      need_compile = True
    if need_compile:
      update_jit = jax.jit(
          functools.partial(
              update,
              num_cats=num_cats,
              delete_mode=delete_mode,
              sequence_boundaries=sequence_boundaries,
              optimiser=optimiser,
              final_num_cats=final_num_cats,
              use_em=use_em,
              fourdof=fourdof,
              cam_focal_length=cam_focal_length,
          )
      )
      need_compile = False
    lr_mul = min(1.0, (num_since_fork + 1) / 20.0)

    # TODO(doersch): hardcoding the LR schedule isn't very smart
    if state.step > num_iters * 0.25:
      lr_mul = lr_mul / 2.0
    if state.step > num_iters * 0.50:
      lr_mul = lr_mul / 2.0
    if state.step > num_iters * 0.75:
      lr_mul = lr_mul / 2.0
    state, metrics = update_jit(state, data + (state.step,), lr_mul)
    loss_curve.append(metrics['loss'])
    loss_moving_average = 0.9 * loss_moving_average + 0.1 * metrics['losses']

    if step % log_every == 0:
      steps_per_sec = log_every / (time.time() - prev_time)
      prev_time = time.time()
      metrics |= {'steps_per_sec': steps_per_sec}
      print(
          {
              k: float(v) if k != 'losses' else list(np.array(v))
              for k, v in metrics.items()
          }
      )

    num_since_fork += 1

  pts_eval_jit = jax.jit(
      functools.partial(
          pts_eval,
          num_cats=num_cats,
          sequence_boundaries=sequence_boundaries,
      )
  )
  sum_error = []

  for i in range(0, separation_tracks.shape[0], 128):
    err, _, _ = pts_eval_jit(
        state,
        np.arange(i, min(separation_tracks.shape[0], i + 128)),
        data[0],
        data[1],
    )
    sum_error.append(np.sum(err, axis=1))

  sum_error = np.concatenate(sum_error, axis=0)

  return {
      'classes': np.array(np.argmin(np.array(sum_error), axis=-1)),
      'sum_error': sum_error,
      'separation_visibility': separation_visibility_dict,
      'separation_tracks': separation_tracks_dict,
      'query_features': query_features,
      'demo_episode_ids': demo_episode_ids,
  }


def construct_fake_causal_state(
    query_features, convert_to_jax=False, channel_multiplier=4
):
  """Constructs a fake TAPIR causal state which can be used to reduce jitting.

  Please not this function is very fragile and only works for the current
  version of tapir. It will likely need to be updated if TAPIR changes, but it
  is helpful for quick iterations.

  Args:
    query_features: Query features which will be used to infer shapes.
    convert_to_jax: Whether to convert it to a jax array (helps prevent
      recompiles)
    channel_multiplier: for compatibility with smaller models

  Returns:
    A causal state.
  """
  num_points = query_features_count(query_features)
  num_resolutions = len(query_features.resolutions)
  dims = 512 * channel_multiplier

  value_shapes = {
      'tapir/~/pips_mlp_mixer/block_1_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_1_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_2_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_2_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_3_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_3_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_4_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_4_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_5_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_5_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_6_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_6_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_7_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_7_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_8_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_8_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_9_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_9_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_10_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_10_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_11_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_11_causal_2': (1, num_points, 2, dims),
      'tapir/~/pips_mlp_mixer/block_causal_1': (1, num_points, 2, 512),
      'tapir/~/pips_mlp_mixer/block_causal_2': (1, num_points, 2, dims),
  }
  fake_ret = {k: np.zeros(v) for k, v in value_shapes.items()}
  fake_ret = [fake_ret] * num_resolutions * 4
  if convert_to_jax:
    fake_ret = jax.tree_map(jnp.array, fake_ret)
  return fake_ret


def _build_online_model_init(frames, query_points, *, tapir_model_kwargs):
  """Build tapir model for initialisation and tracking.

  Args:
    frames:
    query_points:
    tapir_model_kwargs:

  Returns:

  Raises:
    <Any>:
  """
  if not tapir_model_kwargs['use_causal_conv']:
    raise ValueError('Online model requires causal TAPIR training.')
  model = tapir_model.TAPIR(**tapir_model_kwargs)
  feature_grids = model.get_feature_grids(
      frames,
      is_training=False,
  )
  query_features = model.get_query_features(
      frames,
      is_training=True,
      query_points=query_points,
      feature_grids=feature_grids,
  )
  return query_features


def _build_online_model_predict(
    frames,
    query_features,
    causal_context,
    *,
    tapir_model_kwargs,
    query_chunk_size=256,
    query_points_in_video=None,
):
  """Compute point tracks and occlusions given frames and query points."""
  if not tapir_model_kwargs['use_causal_conv']:
    raise ValueError('Online model requires causal TAPIR training.')
  model = tapir_model.TAPIR(**tapir_model_kwargs)
  feature_grids = model.get_feature_grids(
      frames,
      is_training=False,
  )
  trajectories = model.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=query_features,
      query_points_in_video=query_points_in_video,
      query_chunk_size=query_chunk_size,
      causal_context=causal_context,
      get_causal_context=True,
  )
  trajectories = dict(trajectories)
  causal_context = trajectories['causal_context']
  del trajectories['causal_context']
  return {k: v[-1] for k, v in trajectories.items()}, causal_context


def build_models(
    checkpoint_path,
    query_chunk_size=256,
):
  """Build tapir model for initialisation and tracking."""
  ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
  params, state = ckpt_state['params'], ckpt_state['state']

  num_params = hk.data_structures.tree_size(params)
  num_bytes = hk.data_structures.tree_bytes(params)
  print('TAPIR model')
  print(
      f'Number of params: {num_params}',
  )
  print(f'Number of bytes: {num_bytes / 1e6:.2f} MB')

  tapir_model_kwargs = dict(
      use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
  )

  online_model_init_fn = functools.partial(
      _build_online_model_init,
      tapir_model_kwargs=tapir_model_kwargs,
  )
  online_init = hk.transform_with_state(online_model_init_fn)
  online_init_apply = jax.jit(online_init.apply, backend='cpu')
  online_model_predict_fn = functools.partial(
      _build_online_model_predict,
      tapir_model_kwargs=tapir_model_kwargs,
      query_chunk_size=query_chunk_size,
  )
  online_predict = hk.transform_with_state(online_model_predict_fn)
  # Jit is broken here unfortunately.
  online_predict_apply_cpu = online_predict.apply
  online_predict_apply_gpu = jax.jit(online_predict.apply, backend='gpu')
  # online_predict_apply = online_predict.apply

  rng = jax.random.PRNGKey(42)
  online_init_apply = functools.partial(
      online_init_apply, params=params, state=state, rng=rng
  )
  online_predict_apply_cpu = functools.partial(
      online_predict_apply_cpu, params=params, state=state, rng=rng
  )
  online_predict_apply_gpu = functools.partial(
      online_predict_apply_gpu, params=params, state=state, rng=rng
  )
  return online_init_apply, online_predict_apply_cpu, online_predict_apply_gpu


def query_features_join(feature_list):
  """Merge a list of query features int a single query features structure."""
  lowres = [x.lowres for x in feature_list]
  hires = [x.hires for x in feature_list]

  joined_features = tapir_model.QueryFeatures(
      lowres=tuple(np.concatenate(x, axis=1) for x in zip(*lowres)),
      hires=tuple(np.concatenate(x, axis=1) for x in zip(*hires)),
      resolutions=feature_list[0].resolutions,
  )
  return joined_features


def query_features_count(features):
  """Number of points within a query features structure."""
  return features.lowres[0].shape[1]


def predictions_to_tracks_visibility(predictions, single_step=True):
  """Extract tracks and visibility from TAPIR predictions.

  Args:
    predictions: Predictions output of TAPIR
    single_step: Whether we are processing a single step or a whole episode.

  Returns:
    * Tracks of shape [num_points, (t), xy]
    * Visibility of shape [num_points, (t)]. Float between 0 and 1.
  """
  tracks = predictions['tracks'][0]
  occlusion = predictions['occlusion'][0]
  expected_dist = predictions['expected_dist'][0]
  if single_step:
    tracks = tracks[:, 0]
    occlusion = occlusion[:, 0]
    expected_dist = expected_dist[:, 0]
  pred_occ = jax.nn.sigmoid(occlusion)
  visibility = (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
  return tracks, visibility


def preprocess_frames(frames: np.ndarray) -> np.ndarray:
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def track_many_points(
    separation_videos,
    demo_episode_ids,
    checkpoint_path,
    frame_stride=4,
    points_per_frame=8,
    point_batch_size=2048,
    sample_box_corners=(0.1, 0.1, 0.9, 0.9),
):
  """Track random points sampled from the videos.

  Args:
    separation_videos: dict of uint8 tensors, keyed by demo_episode_ids.
    demo_episode_ids: demo episode ids.
    checkpoint_path: path to TAPIR checkpoint file
    frame_stride: sample points from one out of every this number of frames
    points_per_frame: from each selected frame, sample this many points
    point_batch_size: how many points to run through tapir simultaneously.
    sample_box_corners: sample within this box.  Format is [x_lower, y_lower,
      x_upper, y_upper], in normalized coordinates.

  Returns:
    a dict with tracked points
  """
  tapir_init, _, tapir_predict = build_models(
      checkpoint_path, query_chunk_size=512
  )
  np.random.seed(42)

  query_features = []
  query_features2 = []
  query_points = []
  tmp_query_points = []

  def merge_struct(query_features, tmp_query_points):
    query_features2.append(query_features_join(query_features))
    query_points.append(
        [
            np.concatenate([x[i] for x in tmp_query_points], axis=0)
            for i in range(3)
        ]
    )

  for sv_idx, sv in enumerate([separation_videos[x] for x in demo_episode_ids]):
    print(f'extracting query features for video {sv_idx}')
    for i in range(0, len(sv), frame_stride):
      x_scl = sample_box_corners[2] - sample_box_corners[0]
      y_scl = sample_box_corners[3] - sample_box_corners[1]
      x_add = sample_box_corners[0]
      y_add = sample_box_corners[1]
      qp = (
          np.random.uniform(0.0, 1.0, [points_per_frame, 3])
          * np.array([0.0, sv.shape[1] * y_scl, sv.shape[2] * x_scl])[None, ...]
          + np.array([0.0, sv.shape[1] * y_add, sv.shape[2] * x_add])[None, ...]
      )
      tmp_query_points.append((
          np.array([sv_idx] * points_per_frame),
          np.array([i] * points_per_frame),
          qp[..., 1:],
      ))

      qf, _ = tapir_init(
          frames=preprocess_frames(sv[None, None, i]),
          query_points=qp[None],
      )
      query_features.append(qf)
      if len(query_features) == point_batch_size // points_per_frame:
        merge_struct(query_features, tmp_query_points)
        query_features = []
        tmp_query_points = []

  print('Done extracting query features')

  num_extra = 0
  if query_features:
    merge_struct(query_features, tmp_query_points)
  out_query_features = query_features_join(query_features2)
  out_query_points = [
      np.concatenate([x[i] for x in query_points], axis=0) for i in range(3)
  ]

  if query_features:
    del query_features2[-1]
    del query_points[-1]
    while len(query_features) < point_batch_size // points_per_frame:
      query_features.append(query_features[-1])
      tmp_query_points.append(tmp_query_points[-1])
      num_extra += points_per_frame
    merge_struct(query_features, tmp_query_points)

  all_separation_tracks = []
  all_separation_visibility = []

  print('Note that TAPIR compilation takes a while.')
  print('You may not see GPU usage immediately.')
  for qf_idx, query_features in enumerate(query_features2):
    print(f'query feature batch {qf_idx}/{len(query_features2)}')
    separation_tracks = []
    separation_visibility = []
    for sv_idx, sv in enumerate(
        [separation_videos[x] for x in demo_episode_ids]
    ):
      print(f'tracking in video {sv_idx}/{len(demo_episode_ids)}')
      causal_state = construct_fake_causal_state(
          query_features,
          convert_to_jax=True,
          channel_multiplier=4,
      )
      for i in range(len(sv)):
        (prediction, causal_state), _ = tapir_predict(
            frames=preprocess_frames(sv[None, None, i]),
            query_features=query_features,
            causal_context=causal_state,
        )

        prediction = jax.tree_map(np.array, prediction)

        res = predictions_to_tracks_visibility(prediction)
        separation_tracks.append(res[0])
        separation_visibility.append(res[1] > 0.5)  # Threshold
    all_separation_visibility.append(np.stack(separation_visibility, axis=1))
    all_separation_tracks.append(np.stack(separation_tracks, axis=1))

  separation_visibility = np.concatenate(all_separation_visibility, axis=0)
  separation_tracks = np.concatenate(all_separation_tracks, axis=0)

  pad_start = separation_tracks.shape[0] - num_extra
  separation_tracks = separation_tracks[:pad_start]
  separation_visibility = separation_visibility[:pad_start]

  separation_video_shapes = [
      separation_videos[x].shape for x in demo_episode_ids
  ]
  bnds = []
  cur = 0
  for shp in separation_video_shapes:
    bnds.append((cur, cur + shp[0]))
    cur += shp[0]

  separation_visibility = {
      k: separation_visibility[:, lb:ub]
      for k, (lb, ub) in zip(demo_episode_ids, bnds)
  }
  separation_tracks = {
      k: separation_tracks[:, lb:ub]
      for k, (lb, ub) in zip(demo_episode_ids, bnds)
  }
  return {
      'separation_visibility': separation_visibility,
      'separation_tracks': separation_tracks,
      'video_shape': {
          x: separation_video_shapes[i] for i, x in enumerate(demo_episode_ids)
      },
      'query_features': jax.tree_map(np.array, out_query_features),
      'demo_episode_ids': demo_episode_ids,
      'query_points': out_query_points,
  }

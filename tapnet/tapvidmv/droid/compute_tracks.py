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

"""Stage 3: 3D point candidate sampling and multi-view visibility labeling."""

# pylint: disable=invalid-name

import os

from absl import app
import config as config_lib
import core.geometry
import core.io
import core.physics
import core.runner
import cv2
from ml_collections import config_flags
import numpy as np


def resize_mask(mask, margin):
  kernel = np.ones((abs(margin), abs(margin)), np.uint8)
  morph = cv2.dilate if margin > 0 else cv2.erode
  return morph(mask.astype(np.uint8), kernel).astype(bool)


def near_edge(u, v, width, height, margin):
  rim = np.minimum.reduce([u, v, width - 1 - u, height - 1 - v]) < margin
  return core.geometry.in_frame(u, v, width, height) & rim


def on_fixed_rim(points, episode, poses, frame, src_cam, margin):
  rim = np.zeros(len(points), dtype=bool)
  for cam_id, cam_data in episode["camera"].items():
    if cam_id in (src_cam, episode["meta"]["wrist_cam_id"]):
      continue
    u, v, z = core.geometry.project_points(
        points, cam_data["K"], poses[cam_id]["T_cam2world"][frame]
    )
    rim |= (z > 0) & near_edge(
        u, v, *cam_data["raw_depth"][frame].shape[::-1], margin
    )
  return rim


def query_frames(episode, poses, pb_renderer, config):
  """Selects query frames that maximize multi-view robot arm visibility."""
  robot = episode["robot"]
  reach = len(robot["joint_positions"])

  seen = np.zeros((len(episode["camera"]), reach))
  for frame in range(reach):
    pb_renderer.update_robot_pose(
        robot["joint_positions"][frame],
        gripper_state=robot["gripper_positions"][frame],
    )
    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      height, width = cam_data["raw_depth"][frame].shape
      drawn = pb_renderer.render_depth(
          poses[cam_id]["T_cam2world"][frame], cam_data["K"], width, height
      )
      seen[view, frame] = resize_mask(
          drawn > 0, -config.tracks.mask_margin
      ).sum()

  worst = (seen / np.maximum(seen.max(axis=1, keepdims=True), 1)).min(axis=0)
  edges = np.linspace(0, reach, config.tracks.num_query_frames + 1).astype(int)
  return np.array(
      [0]
      + [
          lo + int(np.argmax(worst[lo:hi]))
          for lo, hi in zip(edges[1:-1], edges[2:])
      ]
  )


def spread_cells(points, min_gap):
  order = np.random.permutation(len(points))
  _, first = np.unique(
      np.round(points[order] / min_gap).astype(np.int64),
      axis=0,
      return_index=True,
  )
  return order[first]


def urdf_gap(depth, u, v, z_pred):
  z = core.geometry.sample_depth(depth, u, v, z_pred)
  return np.where(z == 0, np.inf, z) - z_pred


def sensor_slack(cam_data, frame, u, v, z_pred, config):
  gap = urdf_gap(cam_data["raw_depth"][frame], u, v, z_pred)
  stereo = (
      z_pred**2
      * config.tracks.disparity_tolerance
      / (cam_data["K"][0, 0] * cam_data["baseline"])
  )
  return gap / np.maximum(config.tracks.sensor_tolerance_floor, stereo)


def link_masks(links):
  return [(link, links == link) for link in np.unique(links).tolist()]


def link_local(points_world, links, robot_id):
  homogeneous = np.hstack([points_world, np.ones_like(points_world[:, :1])]).T
  local = np.zeros_like(homogeneous)
  for link, on_link in link_masks(links):
    local[:, on_link] = (
        np.linalg.inv(core.physics.link_transform(robot_id, link))
        @ homogeneous[:, on_link]
    )
  return local


def find_robot_candidates(episode, poses, pb_renderer, queries, config):
  """Samples link-local candidate points on the visible URDF robot surface."""
  robot = episode["robot"]

  local, links, query_view, query_frame = [], [], [], []
  for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
    for frame in queries:
      pb_renderer.update_robot_pose(
          robot["joint_positions"][frame],
          gripper_state=robot["gripper_positions"][frame],
      )

      K = cam_data["K"]
      height, width = cam_data["raw_depth"][frame].shape
      T_cam2world = poses[src_cam]["T_cam2world"][frame]

      link_ids, urdf_depth = pb_renderer.render_segmentation(
          T_cam2world, K, width, height
      )
      vs, us = np.where(resize_mask(urdf_depth > 0, -config.tracks.mask_margin))

      z = urdf_depth[vs, us]
      sensor = sensor_slack(cam_data, frame, us, vs, z, config)
      lit = (sensor >= -1) & ~near_edge(
          us, vs, width, height, config.tracks.edge_margin
      )

      surface = core.geometry.unproject_pixels(us, vs, z, K, T_cam2world)
      keep = lit & ~on_fixed_rim(
          surface, episode, poses, frame, src_cam, config.tracks.edge_margin
      )
      surface = surface[keep]
      cell = spread_cells(surface, config.tracks.min_gap)
      cell_links = link_ids[vs, us][keep][cell]

      local.append(link_local(surface[cell], cell_links, pb_renderer.robot_id))
      links.append(cell_links)
      query_view.append(np.full(len(cell), view, dtype=np.int8))
      query_frame.append(np.full(len(cell), frame, dtype=np.int32))

  return (
      np.concatenate(local, axis=1),
      np.concatenate(links),
      np.concatenate(query_view),
      np.concatenate(query_frame),
  )


def carry_robot(local, links, robot, pb_renderer):
  """Propagates link-local robot surface points across all episode frames."""
  n_frames = len(robot["joint_positions"])
  carried = np.zeros((n_frames, local.shape[1], 3), dtype=np.float32)
  on_links = link_masks(links)
  for t in range(n_frames):
    pb_renderer.update_robot_pose(
        robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t]
    )
    for link, on_link in on_links:
      carried[t, on_link] = (
          core.physics.link_transform(pb_renderer.robot_id, link)
          @ local[:, on_link]
      )[:3].T
  return carried


def depth_steps(depth, window=5):
  kernel = np.ones((window, window), np.uint8)
  far = cv2.dilate(depth, kernel)
  near = -cv2.dilate(np.where(depth > 0, -depth, -np.inf), kernel)
  return np.where(far > 0, far - near, np.inf)


def find_static_candidates(episode, poses, pb_renderer, queries, config):
  """Samples cross-camera consistent static background 3D point candidates."""
  robot = episode["robot"]
  match_radius, max_depth = config.tracks.match_radius, config.tracks.max_depth
  max_step = config.tracks.max_edge_step

  points_3d, query_view, query_frame = [], [], []
  for frame in queries:
    pb_renderer.update_robot_pose(
        robot["joint_positions"][frame],
        gripper_state=robot["gripper_positions"][frame],
    )

    drawn, steps = {}, {}
    for cam_id, cam_data in episode["camera"].items():
      height, width = cam_data["raw_depth"][frame].shape
      drawn[cam_id] = pb_renderer.render_depth(
          poses[cam_id]["T_cam2world"][frame], cam_data["K"], width, height
      )
      steps[cam_id] = depth_steps(cam_data["raw_depth"][frame])

    for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
      depth = cam_data["raw_depth"][frame]
      K, T_cam2world = cam_data["K"], poses[src_cam]["T_cam2world"][frame]

      on_env = (
          ~resize_mask(drawn[src_cam] > 0, config.tracks.mask_margin)
          & (depth > 0)
          & (depth <= max_depth)
      )
      vs, us = np.where(on_env & (steps[src_cam] <= max_step))

      points = core.geometry.unproject_pixels(
          us, vs, depth[vs, us], K, T_cam2world
      )

      confirmed = np.zeros(len(points), dtype=bool)
      doubted = near_edge(us, vs, *depth.shape[::-1], config.tracks.edge_margin)
      doubted |= on_fixed_rim(
          points, episode, poses, frame, src_cam, config.tracks.edge_margin
      )
      for other_cam, other_data in episode["camera"].items():
        if other_cam == src_cam:
          continue
        u, v, z = core.geometry.project_points(
            points, other_data["K"], poses[other_cam]["T_cam2world"][frame]
        )
        z_other = core.geometry.sample_depth(
            other_data["raw_depth"][frame], u, v, z
        )
        step = core.geometry.sample_depth(steps[other_cam], u, v, z)
        behind_arm = core.geometry.sample_depth(drawn[other_cam], u, v, z) > 0

        speaks = np.isfinite(z_other) & (z_other > 0) & ~behind_arm
        agrees = np.abs(z_other - z) < match_radius
        confirmed |= speaks & agrees & (z_other <= max_depth)
        doubted |= speaks & (~agrees | ~(step <= max_step))

      cell = spread_cells(points[confirmed & ~doubted], config.tracks.min_gap)
      points_3d.append(points[confirmed & ~doubted][cell])
      query_view.append(np.full(len(cell), view, dtype=np.int8))
      query_frame.append(np.full(len(cell), frame, dtype=np.int32))

  return (
      np.concatenate(points_3d),
      np.concatenate(query_view),
      np.concatenate(query_frame),
  )


def project_tracks(robot_3d, static_3d, episode, poses, pb_renderer, config):
  """Projects 3D tracks into all views and labels per-frame visibility."""
  robot = episode["robot"]
  n_frames, n_robot, _ = robot_3d.shape
  n_points = n_robot + len(static_3d)
  n_views = len(episode["camera"])

  margin = np.zeros((n_views, n_frames, n_points), dtype=np.float32)
  inside = np.zeros((n_views, n_frames, n_points), dtype=bool)
  slack = np.zeros((n_views, n_frames, n_points), dtype=np.float32)
  masks = [
      np.zeros((n_frames, *cam_data["raw_depth"][0].shape), dtype=bool)
      for cam_data in episode["camera"].values()
  ]

  for t in range(n_frames):
    pb_renderer.update_robot_pose(
        robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t]
    )
    points_t = np.concatenate([robot_3d[t], static_3d])

    drawn = [
        pb_renderer.render_depth(
            poses[cam_id]["T_cam2world"][t],
            cam_data["K"],
            *cam_data["raw_depth"][t].shape[::-1],
        )
        for cam_id, cam_data in episode["camera"].items()
    ]

    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      K = cam_data["K"]
      T_cam2world = poses[cam_id]["T_cam2world"][t]
      urdf_depth = drawn[view]
      masks[view][t] = urdf_depth > 0

      height, width = cam_data["raw_depth"][t].shape
      u, v, z_pred = core.geometry.project_points(points_t, K, T_cam2world)
      inside[view, t] = core.geometry.in_frame(u, v, width, height) & (
          z_pred > 0
      )

      urdf = urdf_gap(urdf_depth, u, v, z_pred)
      sensor = sensor_slack(cam_data, t, u, v, z_pred, config)

      margin[view, t] = np.fmin(
          urdf / config.tracks.urdf_tolerance,
          np.where(np.isinf(sensor), np.nan, sensor),
      )
      slack[view, t] = sensor

  return settle(latch(margin, config.tracks.hysteresis), inside), slack, masks


def latch(margin, band):
  hidden, shown = margin < -1 - band, margin > -1 + band

  vis = np.empty(margin.shape, dtype=bool)
  vis[:, 0] = ~(margin[:, 0] < -1)
  for t in range(1, margin.shape[1]):
    vis[:, t] = np.where(
        hidden[:, t], False, np.where(shown[:, t], True, vis[:, t - 1])
    )
  return vis


def settle(vis, inside):
  settled = vis
  for _ in range(4):
    middle = settled[:, 1:-1]
    alone = (middle != settled[:, :-2]) & (middle != settled[:, 2:])
    if not alone.any():
      break
    settled = settled.copy()
    settled[:, 1:-1] = np.where(alone, ~middle, middle)
  return settled & inside


def never_seen_through(slack, max_seen_through):
  seen_through = np.isfinite(slack) & (slack > 1)
  clear_line = np.isfinite(slack) & (slack >= -1)
  return ~(
      seen_through.sum(axis=1) / np.maximum(clear_line.sum(axis=1), 1)
      > max_seen_through
  ).any(axis=0)


def out_of_reach(points_3d, episode, pb_renderer, clearance):
  """Filters static points that come within clearance of the moving gripper."""
  robot = episode["robot"]

  closest = np.full(len(points_3d), np.inf, dtype=np.float32)
  for t in range(len(robot["joint_positions"])):
    pb_renderer.update_robot_pose(
        robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t]
    )
    links = np.array([
        core.physics.link_transform(pb_renderer.robot_id, link)[:3, 3]
        for link in pb_renderer.gripper_links
    ])
    closest = np.minimum(
        closest,
        np.linalg.norm(links[:, None] - points_3d[None], axis=-1).min(axis=0),
    )

  return closest > clearance


def export_tracks(episode, tracks, export_root):
  """Writes 3D tracks, visibility masks, and rendered robot masks to disk."""
  episode_id = episode["meta"]["episode_id"]
  episode_dir = os.path.join(export_root, episode_id)
  os.makedirs(episode_dir, exist_ok=True)

  for view, cam_id in enumerate(episode["camera"]):
    cam_dir = os.path.join(episode_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(cam_dir, "robot_mask.npz"), mask=tracks["masks"][view]
    )

  np.savez_compressed(
      os.path.join(episode_dir, "tracks.npz"),
      tracks_3d=tracks["tracks_3d"].swapaxes(0, 1),
      vis=tracks["vis"].swapaxes(1, 2),
      query_frame=tracks["query_frame"].astype(np.int16),
      query_view=tracks["query_view"].astype(np.int8),
      n_robot=np.int32(tracks["n_robot"]),
      n_static=np.int32(tracks["n_static"]),
  )


def find_candidates(episode, poses, pb_renderer, queries, config):
  """Finds robot and static 3D track candidates for an episode."""
  local, links, robot_view, robot_frame = find_robot_candidates(
      episode, poses, pb_renderer, queries, config
  )
  static_3d, static_view, static_frame = find_static_candidates(
      episode, poses, pb_renderer, queries, config
  )
  keep = out_of_reach(
      static_3d, episode, pb_renderer, config.tracks.gripper_clearance
  )
  return {
      "robot_3d": carry_robot(local, links, episode["robot"], pb_renderer),
      "static_3d": static_3d[keep],
      "query_frame": np.concatenate([robot_frame, static_frame[keep]]),
      "query_view": np.concatenate([robot_view, static_view[keep]]),
  }


def label_tracks(tracks, episode, poses, pb_renderer, config):
  """Filters candidate tracks and computes final visibility and foreground masks."""
  robot_3d, static_3d = tracks["robot_3d"], tracks["static_3d"]
  query_frame, query_view = tracks["query_frame"], tracks["query_view"]
  n_robot = robot_3d.shape[1]
  vis, slack, masks = project_tracks(
      robot_3d, static_3d, episode, poses, pb_renderer, config
  )

  keep = vis[query_view, query_frame, np.arange(len(query_view))]
  keep[n_robot:] &= never_seen_through(
      slack[:, :, n_robot:], config.tracks.max_seen_through
  )

  keep_robot, keep_static = keep[:n_robot], keep[n_robot:]
  n_robot, n_static = int(keep_robot.sum()), int(keep_static.sum())
  print(
      f"  {n_robot + n_static} points: {n_robot} robot, {n_static} static | per"
      f" view {np.bincount(query_view[keep])}"
  )

  tracks_3d = np.concatenate(
      [
          robot_3d[:, keep_robot],
          np.broadcast_to(static_3d[keep_static], (len(robot_3d), n_static, 3)),
      ],
      axis=1,
  )
  return {
      "tracks_3d": tracks_3d,
      "vis": vis[:, :, keep],
      "query_frame": query_frame[keep],
      "query_view": query_view[keep],
      "n_robot": n_robot,
      "n_static": n_static,
      "masks": masks,
  }


def check_robot_loss(metrics, max_robot_loss):
  over = {
      k: v
      for k, v in metrics.items()
      if k.startswith("robot_loss_") and (v is None or v > max_robot_loss)
  }
  if over:
    raise core.runner.SkipEpisode(f"robot_loss over {max_robot_loss} m: {over}")


def process_episode(episode_id, pb_renderer, config):
  """Runs Stage 3 3D track extraction and visibility labeling for one episode."""
  check_robot_loss(
      core.io.load_metrics(episode_id, config.paths.extrinsics),
      config.tracks.max_robot_loss,
  )
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)

  queries = query_frames(episode, poses, pb_renderer, config)
  candidates = find_candidates(episode, poses, pb_renderer, queries, config)
  tracks = label_tracks(candidates, episode, poses, pb_renderer, config)

  export_tracks(episode, tracks, config.paths.tracks)


def main(_):
  config = config_flag.value
  pb_renderer = core.physics.PyBulletRenderer(
      config.paths.urdf, gpu=config.render.gpu
  )

  available = core.runner.finished(config.paths.extrinsics, "metrics.json")
  target = core.runner.select_episodes(available, config)
  done = {
      e
      for e in target
      if os.path.exists(os.path.join(config.paths.tracks, e, "tracks.npz"))
  }

  def run_one(episode_id):
    process_episode(episode_id, pb_renderer, config)

  core.runner.run_episodes(target, run_one, done=done, stage="Stage 3")


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config_lib.__file__)
  app.run(main)

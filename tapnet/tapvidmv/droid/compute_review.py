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

"""Exports Rerun (.rrd) recordings for inspecting 3D tracks and camera views."""

# pylint: disable=invalid-name

import os

from absl import app
import config as config_lib
import core.geometry
import core.io
import core.runner
import core.tracks
import core.visualization
from ml_collections import config_flags
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

JPEG_QUALITY = 95

OUTSIDE = [150, 150, 150]
INSPECT = [255, 40, 235]

TRACK_RADIUS_M = 0.003
INSPECT_RADIUS_M = 0.006
RAY_RADIUS_M = 0.0015


class Review:
  """Precomputed episode tracks, queries, and camera geometry for Rerun."""

  def __init__(self, episode, poses, tracks):
    self.tracks_3d = tracks["tracks_3d"]
    self.vis = tracks["vis"]
    self.n_robot = tracks["n_robot"]
    self.query_frame = tracks["query_frame"]
    self.query_view = tracks["query_view"]
    self.is_robot = np.arange(self.tracks_3d.shape[1]) < self.n_robot
    self.track_colors = core.visualization.occlusion_colors(self.is_robot, True)
    uv = core.tracks.project_views(self.tracks_3d, episode, poses)
    self.queries = core.tracks.build_queries(
        uv, self.query_view, self.query_frame
    )

    K, T_cam2world, image_wh = [], [], []
    for cam_id, cam_data in episode["camera"].items():
      height, width = cam_data["raw_depth"][0].shape
      K.append(cam_data["K"])
      T_cam2world.append(poses[cam_id]["T_cam2world"])
      image_wh.append([width, height])

    self.K = np.asarray(K)
    self.T_cam2world = np.asarray(T_cam2world)
    self.image_wh = np.asarray(image_wh, dtype=np.int32)

  @property
  def n_frames(self):
    return len(self.tracks_3d)

  @property
  def n_points(self):
    return self.tracks_3d.shape[1]

  @property
  def n_views(self):
    return len(self.K)


def inspect_tracks(review, n_inspect):
  return sorted(
      np.random.choice(
          review.n_points, min(n_inspect, review.n_points), replace=False
      ).tolist()
  )


def verdict(visible, inside, z, at_query):
  status = "visible" if visible else "hidden"
  if z <= 0:
    status = "behind"
  elif not inside:
    status = "off-frame"
  if visible and not inside:
    status += "!"
  return status + "*" if at_query else status


def frame_status(review, view, t, track):
  u, v, z = core.geometry.project_points(
      review.tracks_3d[t, track][None],
      review.K[view],
      review.T_cam2world[view, t],
  )
  return bool(
      (core.geometry.in_frame(u, v, *review.image_wh[view]) & (z > 0))[0]
  ), float(z[0])


def ray_color(review, view, t, track, inside):
  if not inside:
    return OUTSIDE
  return core.visualization.occlusion_colors(
      track < review.n_robot, review.vis[view, t, track]
  )


def query_cross(review, track):
  x, y = review.queries[track, :2] + 0.5
  return [[[x - 7, y], [x + 7, y]], [[x, y - 7], [x, y + 7]]]


def depth_cloud(depth, img_rgb, K, T_cam2world, stride):
  strided = depth[::stride, ::stride]
  v, u = np.nonzero(strided > 0)
  v, u = v * stride, u * stride
  return (
      core.geometry.unproject_pixels(u, v, depth[v, u], K, T_cam2world),
      img_rgb[v, u],
  )


def blueprint(review, inspect, episode_id, fps):
  center = review.tracks_3d[0, inspect[0]]
  extent = float(
      np.linalg.norm(np.ptp(review.tracks_3d.reshape(-1, 3), axis=0))
  )

  return rrb.Blueprint(
      rrb.Vertical(
          rrb.Spatial3DView(
              name=f"3D tracks — {episode_id}",
              origin="/",
              line_grid=False,
              background=rrb.Background(kind="GradientDark"),
              contents=[
                  "+ /scene/**",
                  "+ /cameras/**",
                  "+ /tracks",
                  "+ /inspect/**",
              ],
              eye_controls=rrb.EyeControls3D(
                  kind="Orbital",
                  look_target=center,
                  position=center
                  + np.array([1.2, -1.4, 0.9]) * max(extent * 0.4, 0.3),
                  eye_up=[0, 0, 1],
              ),
              overrides={
                  f"/inspect/{track}": rrb.EntityBehavior(visible=False)
                  for track in inspect[1:]
              },
          ),
          rrb.Horizontal(*[
              rrb.Spatial2DView(
                  name=f"Camera {view}",
                  origin=f"/views/{view}",
                  visual_bounds=rrb.VisualBounds2D(
                      x_range=[0, int(wh[0])], y_range=[0, int(wh[1])]
                  ),
              )
              for view, wh in enumerate(review.image_wh)
          ]),
          row_shares=[0.70, 0.30],
      ),
      rrb.TimePanel(
          state="collapsed",
          timeline="frame",
          fps=fps,
          play_state="paused",
          loop_mode="all",
      ),
      rrb.BlueprintPanel(state="collapsed"),
      rrb.SelectionPanel(state="collapsed"),
      collapse_panels=False,
  )


def log_cameras(rec, review, episode, config):
  """Logs RGB views, pinhole intrinsics, camera poses, and unprojected clouds."""
  rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

  cameras = list(episode["camera"].values())
  for view in range(review.n_views):
    K = review.K[view]
    rec.log(
        f"/cameras/{view}/pose/image",
        rr.Pinhole(
            focal_length=[K[0, 0], K[1, 1]],
            principal_point=[K[0, 2] + 0.5, K[1, 2] + 0.5],
            resolution=review.image_wh[view],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.08,
        ),
        static=True,
    )

  n_scene_points, n_off_image = 0, 0
  for t in range(review.n_frames):
    rec.set_time("frame", sequence=t)
    for view, cam_data in enumerate(cameras):
      K = review.K[view]
      img_rgb = cam_data["video_rgb"][t]
      T_cam2world = review.T_cam2world[view, t]

      rec.log(
          f"/views/{view}/rgb",
          rr.Image(img_rgb).compress(jpeg_quality=JPEG_QUALITY),
      )
      rec.log(
          f"/cameras/{view}/pose",
          rr.Transform3D(
              translation=T_cam2world[:3, 3], mat3x3=T_cam2world[:3, :3]
          ),
      )

      u, v, z = core.geometry.project_points(
          review.tracks_3d[t], K, T_cam2world
      )
      inside = core.geometry.in_frame(u, v, *review.image_wh[view]) & (z > 0)
      visible = review.vis[view, t]
      n_off_image += int((visible & ~inside).sum())
      rec.log(
          f"/views/{view}/tracks",
          rr.Points2D(
              np.stack([u[inside], v[inside]], axis=1) + 0.5,
              colors=core.visualization.occlusion_colors(
                  review.is_robot[inside], visible[inside]
              ),
              radii=rr.Radius.ui_points(3),
              labels=[str(point) for point in np.flatnonzero(inside)],
              show_labels=False,
              draw_order=20,
          ),
      )

      points_world, colors = depth_cloud(
          cam_data["raw_depth"][t], img_rgb, K, T_cam2world, config.depth_stride
      )
      rec.log(
          f"/scene/{view}",
          rr.Points3D(points_world, colors=colors, radii=config.scene_radius),
      )
      n_scene_points += len(points_world)

    rec.flush(timeout_sec=120)

  return n_scene_points, n_off_image


def log_tracks(rec, review):
  colors = review.track_colors
  labels = [str(point) for point in range(review.n_points)]

  for t, points_world in enumerate(review.tracks_3d):
    rec.set_time("frame", sequence=t)
    rec.log(
        "/tracks",
        rr.Points3D(
            points_world,
            colors=colors,
            radii=TRACK_RADIUS_M,
            labels=labels,
            show_labels=False,
        ),
    )


def log_inspect(rec, review, inspect):
  """Logs 3D sight-line rays and per-camera status labels for inspected tracks."""
  centers = review.T_cam2world[:, :, :3, 3]

  for track in inspect:
    tracks_3d = review.tracks_3d[:, track]
    kind = "robot" if track < review.n_robot else "static"

    for t, point_world in enumerate(tracks_3d):
      rec.set_time("frame", sequence=t)
      status = [
          frame_status(review, view, t, track) for view in range(review.n_views)
      ]
      calls = [
          f"cam{view}"
          f" {verdict(bool(review.vis[view, t, track]), *status[view], t == review.query_frame[track] and view == review.query_view[track])}"
          for view in range(review.n_views)
      ]
      rec.log(
          f"/inspect/{track}/point",
          rr.Points3D(
              point_world[None],
              colors=INSPECT,
              radii=INSPECT_RADIUS_M,
              labels=[f"{track} ({kind}) | " + " | ".join(calls)],
              show_labels=True,
          ),
      )
      rec.log(
          f"/inspect/{track}/rays",
          rr.LineStrips3D(
              [
                  [point_world, centers[view, t]]
                  for view in range(review.n_views)
              ],
              colors=[
                  ray_color(review, view, t, track, status[view][0])
                  for view in range(review.n_views)
              ],
              radii=RAY_RADIUS_M,
          ),
      )


def log_inspect_views(rec, review, inspect):
  """Logs 2D projections and query crosshairs for inspected tracks."""
  inspect = np.asarray(inspect)
  query_frame = review.query_frame[inspect]
  query_view = review.query_view[inspect]
  labels = [str(track) for track in inspect]

  for view in range(review.n_views):
    width, height = review.image_wh[view]
    for t in range(review.n_frames):
      rec.set_time("frame", sequence=t)
      u, v, z = core.geometry.project_points(
          review.tracks_3d[t, inspect],
          review.K[view],
          review.T_cam2world[view, t],
      )
      inside = core.geometry.in_frame(u, v, width, height) & (z > 0)
      visible = review.vis[view, t, inspect]

      rec.log(
          f"/views/{view}/inspect",
          rr.Points2D(
              np.stack([u[inside], v[inside]], axis=1) + 0.5,
              colors=core.visualization.occlusion_colors(
                  review.is_robot[inspect][inside], visible[inside]
              ),
              radii=rr.Radius.ui_points(3),
              labels=[labels[i] for i in np.flatnonzero(inside)],
              show_labels=True,
              draw_order=21,
          ),
      )
      born = (query_frame == t) & (query_view == view)
      crosses = []
      for i in np.flatnonzero(born):
        crosses.extend(query_cross(review, int(inspect[i])))
      rec.log(
          f"/views/{view}/query",
          rr.LineStrips2D(
              crosses,
              colors=INSPECT,
              radii=0.8,
              draw_order=23,
          ),
      )


def build_recording(episode, review, episode_id, review_root, config):
  """Builds and atomically writes the Rerun recording file for one episode."""
  rrd = os.path.join(review_root, f"{episode_id}.rrd")
  os.makedirs(os.path.dirname(rrd), exist_ok=True)
  staging = rrd + ".partial"
  inspect = inspect_tracks(review, config.n_inspect)

  rec = rr.RecordingStream(
      episode_id.replace("+", "-"), recording_id=episode_id
  )
  try:
    rec.save(
        staging,
        default_blueprint=blueprint(review, inspect, episode_id, config.fps),
    )
    log_tracks(rec, review)
    log_inspect(rec, review, inspect)
    log_inspect_views(rec, review, inspect)
    rec.flush(timeout_sec=120)
    n_scene_points, n_off_image = log_cameras(rec, review, episode, config)
  finally:
    rec.disconnect()
  os.replace(staging, rrd)

  print(
      f"  {review.n_points} tracks ({review.n_robot} robot,"
      f" {review.n_points - review.n_robot} static) | {review.n_frames} frames"
      f" | {review.n_views} views | {n_scene_points / 1e6:.1f}M scene points |"
      f" annotated visible {100 * review.vis.mean():.0f}% | visible off-image"
      f" {n_off_image} | inspecting {len(inspect)} tracks, showing {inspect[0]}"
      f" | {os.path.getsize(rrd) / 1024**2:.0f} MiB"
  )


def process_episode(episode_id, config):
  """Loads episode artifacts and exports a Rerun (.rrd) recording."""
  episode = core.io.load_depth_data(
      episode_id, config.paths.depth, load_video=True
  )
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  tracks = core.io.load_track_data(episode_id, config.paths.tracks)
  tracks = core.tracks.sample_tracks(tracks, config.tracks.points_per_class)

  review = Review(episode, poses, tracks)
  build_recording(
      episode, review, episode_id, config.paths.review, config.review
  )


def main(_):
  config = config_flag.value

  available = core.runner.finished(config.paths.tracks, "tracks.npz")
  target = core.runner.select_episodes(available, config)
  done = {
      e
      for e in target
      if os.path.exists(os.path.join(config.paths.review, f"{e}.rrd"))
  }

  def run_one(episode_id):
    process_episode(episode_id, config)

  core.runner.run_episodes(target, run_one, done=done, stage="Review")


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config_lib.__file__)
  app.run(main)

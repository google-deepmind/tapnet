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

"""Track subsampling, multi-view projection, and query formatting helpers."""

import core.geometry
import numpy as np


def take_farthest(picked, home, group, quota):
  if quota <= 0 or group.size == 0:
    return picked
  return np.concatenate([
      picked,
      group[
          core.geometry.farthest_points(home[group], quota, seeds=home[picked])
      ],
  ])


def sample_tracks(tracks, points_per_class):
  """Subsamples robot and static tracks via farthest-point sampling per view."""
  if (
      points_per_class is None
      or points_per_class <= 0
      or tracks["query_view"].size == 0
  ):
    return tracks

  home = tracks["tracks_3d"][0]
  query_view, query_frame = tracks["query_view"], tracks["query_frame"]
  n_robot = tracks["n_robot"]
  is_robot = np.arange(len(query_view)) < n_robot

  on_arm = in_scene = np.empty(0, dtype=int)
  for frame in np.unique(query_frame):
    for view in range(int(query_view.max()) + 1):
      born_here = (query_view == view) & (query_frame == frame)
      on_arm = take_farthest(
          on_arm, home, np.flatnonzero(born_here & is_robot), points_per_class
      )
      in_scene = take_farthest(
          in_scene,
          home,
          np.flatnonzero(born_here & ~is_robot),
          points_per_class,
      )

  idx = np.sort(np.concatenate([on_arm, in_scene]))
  n_robot = int((idx < n_robot).sum())
  return {
      **tracks,
      "tracks_3d": tracks["tracks_3d"][:, idx],
      "vis": tracks["vis"][:, :, idx],
      "query_view": query_view[idx],
      "query_frame": query_frame[idx],
      "n_robot": n_robot,
      "n_static": len(idx) - n_robot,
  }


def project_views(tracks_3d, episode, poses):
  return np.stack([
      np.stack(
          core.geometry.project_points(
              tracks_3d, cam["K"], poses[cam_id]["T_cam2world"]
          )[:2],
          axis=-1,
      )
      for cam_id, cam in episode["camera"].items()
  ])


def build_queries(uv, query_view, query_frame):
  xy = np.round(uv[query_view, query_frame, np.arange(uv.shape[2])])
  return np.concatenate(
      [xy, query_frame[:, None], query_view[:, None]], axis=1
  ).astype(np.float32)

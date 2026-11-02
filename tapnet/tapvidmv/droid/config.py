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

"""Default configuration for the DROID multi-view 3D tracking pipeline."""

import os

import ml_collections


def get_config():
  """Returns the default pipeline ConfigDict."""
  config = ml_collections.ConfigDict()

  repo = os.path.dirname(os.path.abspath(__file__))
  data = os.path.join(repo, "data")
  output = os.path.join(data, "output", "droid")

  config.paths = ml_collections.ConfigDict()
  config.paths.meta = os.path.join(data, "meta", "1.0.1")
  config.paths.raw = os.path.join(
      data, "input", "robotics", "droid_raw", "1.0.1"
  )
  config.paths.urdf = os.path.join(
      repo, "assets", "franka_description", "franka_panda_robotiq_2f85_og.urdf"
  )
  config.paths.depth = os.path.join(output, "depth")
  config.paths.extrinsics = os.path.join(output, "extrinsics")
  config.paths.tracks = os.path.join(output, "tracks")
  config.paths.review = os.path.join(output, "review")

  config.urls = ml_collections.ConfigDict()
  config.urls.meta = "https://huggingface.co/KarlP/droid/resolve/main"

  config.runner = ml_collections.ConfigDict()
  config.runner.rank = 0
  config.runner.world_size = 1
  config.runner.limit = -1

  config.render = ml_collections.ConfigDict()
  config.render.gpu = True
  config.render.workers = 6

  config.depth = ml_collections.ConfigDict()
  config.depth.min_frames = 48
  config.depth.max_frames = 250
  config.depth.conf_thresh = 0.95
  config.depth.max_depth_thresh = 0.15
  config.depth.gripper_open_thresh = 0.05
  config.depth.mask_area_min = 0.02
  config.depth.mask_area_max = 0.45
  config.depth.consensus_thresh = 0.5

  config.extrinsics = ml_collections.ConfigDict()
  config.extrinsics.lr = 0.001
  config.extrinsics.n_steps = 500
  config.extrinsics.n_points = 2000
  config.extrinsics.chamfer_match_radius = 0.05
  config.extrinsics.max_depth = 1.5
  config.extrinsics.chamfer_px_radius = 3.0
  config.extrinsics.robot_weight = 0.6
  config.extrinsics.disparity_truncation = 100.0

  config.tracks = ml_collections.ConfigDict()
  config.tracks.max_robot_loss = 0.03
  config.tracks.num_query_frames = 2
  config.tracks.points_per_class = 10
  config.tracks.min_gap = 0.005
  config.tracks.match_radius = 0.02
  config.tracks.max_edge_step = 0.05
  config.tracks.max_depth = 2.0
  config.tracks.mask_margin = 10
  config.tracks.edge_margin = 64
  config.tracks.urdf_tolerance = 0.006
  config.tracks.sensor_tolerance_floor = 0.025
  config.tracks.disparity_tolerance = 3.0
  config.tracks.hysteresis = 0.5
  config.tracks.max_seen_through = 0.02
  config.tracks.gripper_clearance = 0.08

  config.review = ml_collections.ConfigDict()
  config.review.depth_stride = 4
  config.review.scene_radius = 0.003
  config.review.n_inspect = 30
  config.review.fps = 30

  return config

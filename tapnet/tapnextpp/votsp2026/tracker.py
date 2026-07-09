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

"""TAPNext++ tracker for VOTSp2026.

Implements the VOT folder protocol: reads frames_color.txt and query_*.txt
from the current directory, runs TAPNext++ to track each queried point, and
writes output_*.txt and output_*_visible.txt.

TAPNext++ is a recurrent online tracker. Tracking begins with query points on
the first frame and proceeds step-by-step via an opaque state object. Queries
starting at different frame offsets are run as independent rollouts (new queries
cannot be added to an existing state).

For each real query point, a small grid of 64 extra support points is placed
within a 32px radius around it (in model-input space). Support points are
co-tracked via shared attention and then discarded — they never appear in
output files. The radius in model-input space gives consistent coverage across
the dataset's wide range of source resolutions (224px–2560px wide).

Configuration (hardcoded in CHECKPOINT_URL / N_SUPPORT / SUPPORT_* constants):
    input_resolution=512, n_support=64, support_mode='local',
    support_radius=32.0, support_radius_space='model'
"""

import collections
import os
import pathlib
import urllib.request

import cv2
import numpy as np
from tapnet.tapnextpp.votsp2026.model import TAPNextPP
import torch
from vot.region import Point
from vot.region.io import parse_region

CHECKPOINT_URL = (
    "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
)
INPUT_RESOLUTION = 512
N_SUPPORT = 64
SUPPORT_MODE = "local"
SUPPORT_RADIUS = 32.0
SUPPORT_RADIUS_SPACE = "model"


def _get_checkpoint() -> str:
  """Download checkpoint from CHECKPOINT_URL if not already cached."""
  cache_dir = pathlib.Path.home() / ".cache" / "tapnextpp"
  cache_dir.mkdir(parents=True, exist_ok=True)
  dest = cache_dir / "tapnextpp_512.ckpt"
  if not dest.exists():
    print(f"Downloading checkpoint from {CHECKPOINT_URL} ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    print(f"Saved to {dest}")
  return str(dest)


def load_model(
    checkpoint_path: str | None = None,
    input_resolution: int = TAPNextPP.MODEL_SIZE,
) -> TAPNextPP:
  """Load TAPNextPP model from checkpoint.

  Args:
    checkpoint_path: Path to checkpoint file.
    input_resolution: Resolution to resize input frames to.

  Returns:
    Loaded TAPNextPP model.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = TAPNextPP.from_checkpoint(
      checkpoint_path, device=device, input_resolution=input_resolution
  )
  print(
      f"TAPNext++ loaded on {device} "
      f"(checkpoint={checkpoint_path}, input_resolution={input_resolution})."
  )
  return model


def grid_support_points(n: int, w: float, h: float) -> np.ndarray:
  """Return n points on an aspect-ratio-matched grid spanning [0,w) x [0,h)."""
  if n <= 0:
    return np.zeros((0, 2), dtype=np.float32)
  cols = max(1, round(float(np.sqrt(n * w / h))))
  rows = max(1, int(np.ceil(n / cols)))
  xs = (np.arange(cols) + 0.5) * (w / cols)
  ys = (np.arange(rows) + 0.5) * (h / rows)
  gx, gy = np.meshgrid(xs, ys)
  pts = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
  return pts[:n]


def local_support_points(
    query_xy: np.ndarray,
    n_per_query: int,
    radius_x: float,
    radius_y: float,
    w: int,
    h: int,
) -> np.ndarray:
  """Return n_per_query points around each query, clamped to [0,w) x [0,h)."""
  if n_per_query <= 0 or len(query_xy) == 0:
    return np.zeros((0, 2), dtype=np.float32)

  all_pts = []
  for qx, qy in query_xy:
    local = grid_support_points(n_per_query, 2 * radius_x, 2 * radius_y)
    local = local - np.array([radius_x, radius_y], dtype=np.float32)
    local = local + np.array([qx, qy], dtype=np.float32)
    local[:, 0] = np.clip(local[:, 0], 0, w - 1)
    local[:, 1] = np.clip(local[:, 1], 0, h - 1)
    all_pts.append(local)
  return np.concatenate(all_pts, axis=0).astype(np.float32)


def main() -> None:
  n_support = N_SUPPORT
  support_mode = SUPPORT_MODE
  support_radius = SUPPORT_RADIUS
  support_radius_space = SUPPORT_RADIUS_SPACE
  input_resolution = INPUT_RESOLUTION
  checkpoint_path = _get_checkpoint()

  frame_files = sorted(
      f
      for f in os.listdir(".")
      if f.startswith("frames_") and f.endswith(".txt")
  )
  assert frame_files, "No frames_*.txt found in current directory"
  with open(frame_files[0]) as fp:
    frame_paths = [line.strip() for line in fp if line.strip()]
  t_total = len(frame_paths)

  query_files = sorted(
      f
      for f in os.listdir(".")
      if f.startswith("query_") and f.endswith(".txt")
  )
  assert query_files, "No query_*.txt found in current directory"
  queries = []
  for qf in query_files:
    oid = qf[len("query_") : -len(".txt")]
    with open(qf) as fp:
      lines = [line.strip() for line in fp if line.strip()]
    offset = int(lines[0])
    region = parse_region(lines[1])
    queries.append((oid, offset, region))

  by_offset = collections.defaultdict(list)
  for oid, offset, region in queries:
    by_offset[offset].append((oid, region))

  model = load_model(checkpoint_path, input_resolution)
  if n_support > 0:
    print(
        f"Using {n_support} support points ({support_mode} mode) per offset"
        " group (not saved to output)."
    )

  results = {}
  visibility = {}

  for offset, group in sorted(by_offset.items()):
    oids = [oid for oid, _ in group]
    n_real = len(oids)

    first_frame_img = None

    def get_first_frame(o=offset):
      nonlocal first_frame_img
      if first_frame_img is None:
        first_frame_img = cv2.imread(frame_paths[o])
      return first_frame_img

    query_xy = []
    for _, region in group:
      if isinstance(region, Point):
        query_xy.append([region.x, region.y])
      else:
        img = get_first_frame()
        h, w = img.shape[:2]
        query_xy.append([w / 2.0, h / 2.0])
    query_xy = np.array(query_xy, dtype=np.float32)

    if n_support > 0:
      img = get_first_frame()
      h, w = img.shape[:2]
      if support_mode == "global":
        support_xy = grid_support_points(n_support, w, h)
      else:
        if support_radius_space == "model":
          radius_x = support_radius * (w / input_resolution)
          radius_y = support_radius * (h / input_resolution)
        else:
          radius_x = radius_y = support_radius
        support_xy = local_support_points(
            query_xy, n_support, radius_x, radius_y, w, h
        )
      query_xy = np.concatenate([query_xy, support_xy], axis=0)

    print(
        f"Tracking {n_real} queries (+{len(query_xy) - n_real} support) "
        f"from offset={offset}, {t_total - offset} frames..."
    )
    traj = {oid: [] for oid in oids}
    vis = {oid: [] for oid in oids}
    tap_state = None

    for i in range(offset, t_total):
      frame = cv2.imread(frame_paths[i])
      if frame is None:
        raise RuntimeError(f"Could not read: {frame_paths[i]}")
      if i == offset:
        positions, visible, tap_state = model.track_frame(
            frame, query_points_xy=query_xy
        )
      else:
        positions, visible, tap_state = model.track_frame(
            frame, state=tap_state
        )
      for j, oid in enumerate(oids):
        traj[oid].append((float(positions[j, 0]), float(positions[j, 1])))
        vis[oid].append(bool(visible[j]))

    for oid in oids:
      results[oid] = [None] * offset + traj[oid]
      visibility[oid] = [None] * offset + vis[oid]

  for oid, positions in results.items():
    with open(f"output_{oid}.txt", "w") as fp:
      for pos in positions:
        fp.write("0\n" if pos is None else f"{pos[0]},{pos[1]}\n")
    with open(f"output_{oid}_visible.txt", "w") as fp:
      for v in visibility[oid]:
        fp.write(("0" if v is None else ("1" if v else "0")) + "\n")

  print(f"Done. Tracked {len(results)} objects over {t_total} frames.")


if __name__ == "__main__":
  main()

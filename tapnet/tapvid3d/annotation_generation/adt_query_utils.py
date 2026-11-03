# Copyright 2026 XiiiiaoT
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

"""Utilities for keeping 2D query points aligned with 3D tracks."""

import numpy as np


def synchronize_query_coordinates(
    queries_xyt: np.ndarray,
    tracks_xyz: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
    tolerance_px: float = 1e-2,
) -> np.ndarray:
  """Reprojects query points that disagree with their 3D track at frame t.

  A query point (x, y, t) must agree with the projection of its corresponding
  camera-coordinate 3D track at frame t. This function repairs stale 2D
  coordinates while preserving query frames and already-consistent points.

  Args:
    queries_xyt: Array with shape (N, 3), storing (x, y, t).
    tracks_xyz: Real-valued array with shape (T, N, 3), storing camera-
      coordinate tracks.
    visibility: Boolean-compatible array with shape (T, N).
    intrinsics: Array [fx, fy, cx, cy].
    tolerance_px: Maximum accepted reprojection discrepancy in pixels.
      The default is larger than the observed float32 projection noise and
      much smaller than the smallest stale-coordinate discrepancy in the
      audited Issue #171 clips.

  Returns:
    A copy of queries_xyt with inconsistent (x, y) coordinates replaced by
    their projections at the existing query frames. The input arrays are not
    modified, and the dtype of queries_xyt is preserved.

  Raises:
    ValueError: If array shapes, query frames, visibility, intrinsics,
      tolerance, or query-frame 3D coordinates are invalid.
  """
  queries = np.array(queries_xyt, copy=True)
  tracks = np.asarray(tracks_xyz)
  visible = np.asarray(visibility, dtype=bool)
  camera_intrinsics = np.asarray(intrinsics, dtype=np.float64)

  if queries.ndim != 2 or queries.shape[1] != 3:
    raise ValueError("queries_xyt must have shape (N, 3).")
  if not np.issubdtype(queries.dtype, np.floating):
    raise ValueError("queries_xyt must have a floating-point dtype.")
  if tracks.ndim != 3 or tracks.shape[2] != 3:
    raise ValueError("tracks_xyz must have shape (T, N, 3).")
  if not np.issubdtype(tracks.dtype, np.number) or np.iscomplexobj(tracks):
    raise ValueError("tracks_xyz must contain real-valued coordinates.")
  if tracks.shape[1] != queries.shape[0]:
    raise ValueError("The number of query points and tracks must match.")
  if visible.shape != tracks.shape[:2]:
    raise ValueError("visibility must have shape (T, N).")
  if camera_intrinsics.shape != (4,):
    raise ValueError("intrinsics must contain [fx, fy, cx, cy].")
  if not np.all(np.isfinite(queries)):
    raise ValueError("queries_xyt must contain only finite values.")
  if not np.all(np.isfinite(camera_intrinsics)):
    raise ValueError("intrinsics must contain only finite values.")
  if not np.isfinite(tolerance_px) or tolerance_px < 0:
    raise ValueError("tolerance_px must be a finite non-negative value.")

  query_frames_float = queries[:, 2]
  query_frames = np.rint(query_frames_float).astype(np.int64)
  if not np.allclose(query_frames_float, query_frames, rtol=0, atol=1e-6):
    raise ValueError("Query frame indices must be integers.")
  if np.any(query_frames < 0) or np.any(query_frames >= tracks.shape[0]):
    raise ValueError("A query frame is outside the track sequence.")

  track_indices = np.arange(queries.shape[0])
  query_visible = visible[query_frames, track_indices]
  if not np.all(query_visible):
    invalid_tracks = np.flatnonzero(~query_visible).tolist()
    raise ValueError(
        "Every query point must be visible at its query frame; "
        f"invisible track indices: {invalid_tracks}."
    )

  query_xyz = tracks[query_frames, track_indices].astype(np.float64)
  if not np.all(np.isfinite(query_xyz)):
    raise ValueError("3D query-frame coordinates must be finite.")
  if np.any(query_xyz[:, 2] <= 0):
    raise ValueError("3D query-frame depth must be positive.")

  fx, fy, cx, cy = camera_intrinsics
  if fx <= 0 or fy <= 0:
    raise ValueError("Focal lengths fx and fy must be positive.")
  projected_xy = np.column_stack((
      fx * query_xyz[:, 0] / query_xyz[:, 2] + cx,
      fy * query_xyz[:, 1] / query_xyz[:, 2] + cy,
  ))
  reprojection_error = np.linalg.norm(
      projected_xy - queries[:, :2].astype(np.float64), axis=1
  )
  inconsistent = reprojection_error > tolerance_px
  queries[inconsistent, :2] = projected_xy[inconsistent]
  return queries

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
"""Regression tests for ADT query-coordinate synchronization (issue #171)."""

import unittest

import numpy as np

from tapnet.tapvid3d.annotation_generation import adt_query_utils


class SynchronizeQueryCoordinatesTest(unittest.TestCase):

  def setUp(self):
    # fx=fy=10, cx=cy=5. Frame 1 projects to pixel coordinates (15, 25).
    self.tracks = np.array([
        [[1.0, 1.0, 1.0]],
        [[2.0, 4.0, 2.0]],
    ])
    self.visibility = np.array([[True], [True]])
    self.intrinsics = np.array([10.0, 10.0, 5.0, 5.0])

  def test_repairs_coordinates_at_query_frame(self):
    queries = np.array([[5.0, 5.0, 1.0]])
    result = adt_query_utils.synchronize_query_coordinates(
        queries, self.tracks, self.visibility, self.intrinsics
    )
    np.testing.assert_allclose(result, [[15.0, 25.0, 1.0]])
    np.testing.assert_array_equal(queries, [[5.0, 5.0, 1.0]])

  def test_repairs_small_error_below_issue_threshold(self):
    # The 1.2 px discrepancy is below the 2.5 px audit threshold, but is
    # still far above the 0.01 px consistency tolerance.
    queries = np.array([[16.2, 25.0, 1.0]])
    result = adt_query_utils.synchronize_query_coordinates(
        queries, self.tracks, self.visibility, self.intrinsics
    )
    np.testing.assert_allclose(result, [[15.0, 25.0, 1.0]])

  def test_preserves_consistent_coordinates(self):
    queries = np.array([[15.0, 25.0, 1.0]])
    result = adt_query_utils.synchronize_query_coordinates(
        queries, self.tracks, self.visibility, self.intrinsics
    )
    np.testing.assert_array_equal(result, queries)

  def test_preserves_float32_dtype_and_query_frame(self):
    queries = np.array([[5.0, 5.0, 1.0]], dtype=np.float32)
    result = adt_query_utils.synchronize_query_coordinates(
        queries, self.tracks, self.visibility, self.intrinsics
    )
    self.assertEqual(result.dtype, np.float32)
    self.assertEqual(result[0, 2], queries[0, 2])
    np.testing.assert_allclose(result[0, :2], [15.0, 25.0], atol=1e-5)

  def test_uses_each_tracks_own_query_frame(self):
    tracks = np.array([
        [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]],
        [[2.0, 4.0, 2.0], [3.0, 3.0, 3.0]],
    ])
    visibility = np.ones((2, 2), dtype=bool)
    queries = np.array([[5.0, 5.0, 1.0], [0.0, 0.0, 0.0]])
    result = adt_query_utils.synchronize_query_coordinates(
        queries, tracks, visibility, self.intrinsics
    )
    np.testing.assert_allclose(result, [[15.0, 25.0, 1.0], [10.0, 15.0, 0.0]])

  def test_does_not_mutate_any_input(self):
    queries = np.array([[5.0, 5.0, 1.0]])
    tracks = self.tracks.copy()
    visibility = self.visibility.copy()
    intrinsics = self.intrinsics.copy()
    original_queries = queries.copy()
    original_tracks = tracks.copy()
    original_visibility = visibility.copy()
    original_intrinsics = intrinsics.copy()

    adt_query_utils.synchronize_query_coordinates(
        queries, tracks, visibility, intrinsics
    )

    np.testing.assert_array_equal(queries, original_queries)
    np.testing.assert_array_equal(tracks, original_tracks)
    np.testing.assert_array_equal(visibility, original_visibility)
    np.testing.assert_array_equal(intrinsics, original_intrinsics)

  def test_preserves_coordinates_within_tolerance(self):
    queries = np.array([[15.005, 25.0, 1.0]])
    result = adt_query_utils.synchronize_query_coordinates(
        queries, self.tracks, self.visibility, self.intrinsics
    )
    np.testing.assert_array_equal(result, queries)

  def test_rejects_invisible_query_frame(self):
    visibility = self.visibility.copy()
    visibility[1, 0] = False
    with self.assertRaisesRegex(ValueError, "must be visible"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 1.0]]),
          self.tracks,
          visibility,
          self.intrinsics,
      )

  def test_rejects_out_of_range_query_frame(self):
    with self.assertRaisesRegex(ValueError, "outside"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 2.0]]),
          self.tracks,
          self.visibility,
          self.intrinsics,
      )

  def test_rejects_non_integer_query_frame(self):
    with self.assertRaisesRegex(ValueError, "must be integers"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 1.5]]),
          self.tracks,
          self.visibility,
          self.intrinsics,
      )

  def test_rejects_non_positive_depth(self):
    tracks = self.tracks.copy()
    tracks[1, 0, 2] = 0.0
    with self.assertRaisesRegex(ValueError, "depth must be positive"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 1.0]]),
          tracks,
          self.visibility,
          self.intrinsics,
      )

  def test_rejects_visibility_shape_mismatch(self):
    with self.assertRaisesRegex(ValueError, "visibility must have shape"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 1.0]]),
          self.tracks,
          np.ones((1, 1), dtype=bool),
          self.intrinsics,
      )

  def test_rejects_non_finite_tolerance(self):
    with self.assertRaisesRegex(ValueError, "finite non-negative"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 1.0]]),
          self.tracks,
          self.visibility,
          self.intrinsics,
          tolerance_px=float("nan"),
      )

  def test_rejects_non_positive_focal_length(self):
    intrinsics = self.intrinsics.copy()
    intrinsics[0] = 0.0
    with self.assertRaisesRegex(ValueError, "Focal lengths"):
      adt_query_utils.synchronize_query_coordinates(
          np.array([[5.0, 5.0, 1.0]]),
          self.tracks,
          self.visibility,
          intrinsics,
      )


if __name__ == "__main__":
  unittest.main()

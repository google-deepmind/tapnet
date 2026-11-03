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

  def test_preserves_consistent_coordinates(self):
    queries = np.array([[15.0, 25.0, 1.0]])
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


if __name__ == "__main__":
  unittest.main()

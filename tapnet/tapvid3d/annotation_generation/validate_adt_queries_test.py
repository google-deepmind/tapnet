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
"""Tests for the ADT query-point preflight validator."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tapnet.tapvid3d.annotation_generation import validate_adt_queries


class ValidateAdtQueriesTest(unittest.TestCase):

  def setUp(self):
    self.queries = np.array([[256.0, 256.0, 1.0]])
    self.tracks = np.array([
        [[1.0, 1.0, 1.0]],
        [[0.0, 0.0, 2.0]],
    ])
    self.visibility = np.array([[True], [True]])

  def _write_npz(self, directory, name, queries=None, visibility=None):
    path = directory / name
    np.savez(
        path,
        queries_xyt=self.queries if queries is None else queries,
        tracks_XYZ=self.tracks,
        visibility=(
            self.visibility if visibility is None else visibility
        ),
    )
    return path

  def test_reports_clean_file(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = self._write_npz(Path(temp_dir), "clean.npz")
      result = validate_adt_queries.validate_npz_file(path)
    self.assertEqual(result["status"], "clean")
    self.assertEqual(result["num_corrections"], 0)

  def test_reports_coordinates_that_would_be_repaired(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      stale_queries = np.array([[5.0, 5.0, 1.0]])
      path = self._write_npz(
          Path(temp_dir), "stale.npz", queries=stale_queries
      )
      result = validate_adt_queries.validate_npz_file(path)
    self.assertEqual(result["status"], "would_repair")
    self.assertEqual(result["num_corrections"], 1)
    self.assertLess(result["max_post_reprojection_error_px"], 1e-4)

  def test_reports_invisible_query_as_error(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      visibility = np.array([[True], [False]])
      path = self._write_npz(
          Path(temp_dir), "invisible.npz", visibility=visibility
      )
      result = validate_adt_queries.validate_npz_file(path)
    self.assertEqual(result["status"], "error")
    self.assertIn("must be visible", result["error"])

  def test_summarizes_all_npz_files_in_directory(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      self._write_npz(directory, "clean.npz")
      self._write_npz(
          directory, "stale.npz", queries=np.array([[5.0, 5.0, 1.0]])
      )
      report = validate_adt_queries.validate_directory(directory)
    self.assertEqual(report["summary"]["files_scanned"], 2)
    self.assertEqual(report["summary"]["clean_files"], 1)
    self.assertEqual(report["summary"]["files_requiring_correction"], 1)
    self.assertEqual(report["summary"]["queries_requiring_correction"], 1)


if __name__ == "__main__":
  unittest.main()

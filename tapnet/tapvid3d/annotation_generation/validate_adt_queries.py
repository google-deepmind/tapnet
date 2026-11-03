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

"""Preflight validation for downloaded TAPVid-3D ADT query-point NPZ files.

This tool checks query-frame visibility and reprojection consistency without
requiring the licensed ADT source sequences or Project Aria dependencies.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tapnet.tapvid3d.annotation_generation import adt_query_utils


FOCAL_LENGTH = 280.0
IMAGE_SIZE = 512
DEFAULT_INTRINSICS = np.array([
    FOCAL_LENGTH,
    FOCAL_LENGTH,
    IMAGE_SIZE / 2,
    IMAGE_SIZE / 2,
])
DEFAULT_TOLERANCE_PX = 1e-2


def validate_npz_file(
    npz_path: Path,
    intrinsics: np.ndarray = DEFAULT_INTRINSICS,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
) -> dict[str, object]:
  """Checks one downloaded ADT NPZ without modifying it."""
  try:
    with np.load(npz_path, allow_pickle=True) as data:
      required_fields = ("queries_xyt", "tracks_XYZ", "visibility")
      missing_fields = [field for field in required_fields if field not in data]
      if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
      queries = np.array(data["queries_xyt"], copy=True)
      tracks = np.array(data["tracks_XYZ"], copy=True)
      visibility = np.array(data["visibility"], copy=True)

    corrected = adt_query_utils.synchronize_query_coordinates(
        queries, tracks, visibility, intrinsics, tolerance_px
    )
    correction_px = np.linalg.norm(
        corrected[:, :2].astype(np.float64)
        - queries[:, :2].astype(np.float64),
        axis=1,
    )
    query_frames = np.rint(corrected[:, 2]).astype(np.int64)
    track_indices = np.arange(corrected.shape[0])
    query_xyz = tracks[query_frames, track_indices].astype(np.float64)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    projected_xy = np.column_stack((
        fx * query_xyz[:, 0] / query_xyz[:, 2] + cx,
        fy * query_xyz[:, 1] / query_xyz[:, 2] + cy,
    ))
    post_error_px = np.linalg.norm(
        projected_xy - corrected[:, :2].astype(np.float64), axis=1
    )
    correction_count = int(np.count_nonzero(correction_px > 0))
    return {
        "file": npz_path.name,
        "status": "would_repair" if correction_count else "clean",
        "num_tracks": int(queries.shape[0]),
        "num_corrections": correction_count,
        "max_correction_px": float(np.max(correction_px, initial=0.0)),
        "max_post_reprojection_error_px": float(
            np.max(post_error_px, initial=0.0)
        ),
    }
  except Exception as exc:  # pylint: disable=broad-except
    return {
        "file": npz_path.name,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def validate_directory(
    input_npz_dir: Path,
    intrinsics: np.ndarray = DEFAULT_INTRINSICS,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
) -> dict[str, object]:
  """Checks every NPZ in a directory and returns a JSON-serializable report."""
  if not input_npz_dir.is_dir():
    raise ValueError(f"Input directory does not exist: {input_npz_dir}")
  npz_files = sorted(input_npz_dir.glob("*.npz"))
  if not npz_files:
    raise ValueError(f"No NPZ files found in: {input_npz_dir}")

  results = [
      validate_npz_file(path, intrinsics, tolerance_px) for path in npz_files
  ]
  summary = {
      "files_scanned": len(results),
      "clean_files": sum(item["status"] == "clean" for item in results),
      "files_requiring_correction": sum(
          item["status"] == "would_repair" for item in results
      ),
      "files_with_errors": sum(item["status"] == "error" for item in results),
      "queries_requiring_correction": sum(
          item.get("num_corrections", 0) for item in results
      ),
      "total_queries": sum(item.get("num_tracks", 0) for item in results),
  }
  return {"summary": summary, "files": results}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input-npz-dir",
      type=Path,
      required=True,
      help="Directory containing downloaded ADT query-point NPZ files.",
  )
  parser.add_argument(
      "--report-json",
      type=Path,
      help="Optional path for writing the full per-file JSON report.",
  )
  parser.add_argument(
      "--tolerance-px",
      type=float,
      default=DEFAULT_TOLERANCE_PX,
      help="Reprojection consistency tolerance in pixels.",
  )
  args = parser.parse_args()

  try:
    report = validate_directory(
        args.input_npz_dir,
        DEFAULT_INTRINSICS,
        args.tolerance_px,
    )
  except ValueError as exc:
    parser.error(str(exc))

  summary_json = json.dumps(report["summary"], indent=2, sort_keys=True)
  print(summary_json)
  if args.report_json:
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Full report written to: {args.report_json}")
  return 1 if report["summary"]["files_with_errors"] else 0


if __name__ == "__main__":
  raise SystemExit(main())

# Copyright 2025 Google LLC
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

"""Compute metrics on the saved outputs of a TAP-3D model."""

from collections.abc import Sequence
import io
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from tapnet.tapvid3d.evaluation import metrics
from tapnet.tapvid3d.splits import tapvid3d_splits
import tqdm


_TAPVID3D_DIR = flags.DEFINE_string(
    'tapvid3d_dir',
    'tapvid3d_dataset/',
    """
    Path to folder that has all the ground truth npy files. Should
    contain subfolders for each data source (adt, pstudio, drivetrack).
    """,
)

_TAPVID3D_PREDICTIONS = flags.DEFINE_string(
    'tapvid3d_predictions',
    'tapvid3d_predictions/',
    """
    Must be a folder that contains a set of *.npz files, each one with
    the predicted 3D trajectories for that video, with filename
    exactly matching the corresponding ground truth file/video example.
    The npy file must have a tensor `tracks_xyz` of shape (T, N, 3) and
    a tensor `visible` of shape where [T, N], where N is the
    number of tracks in the video, and T is the number of frames.
    """,
)

_USE_MINIVAL = flags.DEFINE_boolean(
    'use_minival',
    True,
    """
    If True, compute metrics on the minival split;
    otherwise uses the full_eval split.
    """,
)

_DEPTH_SCALINGS = flags.DEFINE_list(
    'depth_scalings',
    ['median'],
    'Depth scaling strategies to use, list of [median, per_trajectory, etc..].',
)

_DATA_SOURCES_TO_EVALUATE = flags.DEFINE_list(
    'data_sources_to_evaluate',
    ['drivetrack', 'adt', 'pstudio'],
    'Which data source subsets to evaluate.',
)

_DEBUG = flags.DEFINE_boolean(
    'debug',
    False,
    'Whether to run in debug mode, downloads only one video.',
)


ZERO_METRICS_DICT = {
    'occlusion_accuracy': np.array([0.0]),
    'pts_within_1': np.array([0.0]),
    'jaccard_1': np.array([0.0]),
    'pts_within_2': np.array([0.0]),
    'jaccard_2': np.array([0.0]),
    'pts_within_4': np.array([0.0]),
    'jaccard_4': np.array([0.0]),
    'pts_within_8': np.array([0.0]),
    'jaccard_8': np.array([0.0]),
    'pts_within_16': np.array([0.0]),
    'jaccard_16': np.array([0.0]),
    'average_jaccard': np.array([0.0]),
    'average_pts_within_thresh': np.array([0.0]),
}


def get_new_hw_with_given_smallest_side_length(
    *, orig_height: int, orig_width: int, smallest_side_length: int = 256
):
  orig_shape = np.array([orig_height, orig_width])
  scaling_factor = smallest_side_length / np.min(orig_shape)
  resized_shape = np.round(orig_shape * scaling_factor)
  return (int(resized_shape[0]), int(resized_shape[1])), scaling_factor


def get_jpeg_byte_hw(jpeg_bytes: bytes):
  with io.BytesIO(jpeg_bytes) as img_bytes:
    img = Image.open(img_bytes)
    img = img.convert('RGB')
  return np.array(img).shape[:2]


def get_average_over_metrics(
    list_of_metrics: list[dict[str, dict[str, np.ndarray]]]
):
  """Average over per video metrics in a nested metrics dictionaries."""
  avg_metrics = {}
  for metric_category in list_of_metrics[0].keys():
    avg_metrics[metric_category] = {}
    for metric_name in list_of_metrics[0][metric_category]:
      avg_metrics[metric_category][metric_name] = np.mean(
          [
              video_metric[metric_category][metric_name]
              for video_metric in list_of_metrics
          ]
      )
  return avg_metrics


def evaluate_data_source(
    npz_filenames: list[str],
    ground_truth_dir: str,
    predictions_dir: str,
    depth_scalings: list[str],
    metric_eval_resolution: int = 256,
) -> dict[str, dict[str, np.ndarray]]:
  """Compute metrics on the set of 3D track predictions.

  This function expects as input two directories: `ground_truth_dir` and
  `predictions_dir`. They should have a structure as follows:

    - ground_truth_dir:
        - video1.npz (with contents+format described in tapvid3d/README.md)
        - video2.npz
        - video3.npz
        - ...
        - videoN.npz

    - predictions_dir:
        - video1.npz (contains two tensors: the predicted `visibility` and
                      `tracks_XYZ`, with shapes described in tapvid3d/README.md)
        - video2.npz
        - video3.npz
        - ...
        - videoN.npz

  `ground_truth_dir` contains npz files with ground truth data, while
  `predictions_dir` should contains npz files with the same file name as their
  corresponding ground truth file in `ground_truth_dir`. npz files can be named
  anything (do not need to be named as video{N}.npz in the example), as long
  as it is consistent across both directories.

  Also note that this directory structure (for `ground_truth_dir`) is what is
  generated by `generate_all.sh` script. The script generates three different
  directories: `tapvid3d_dataset/adt`, `tapvid3d_dataset/pstudio`,
  `tapvid3d_dataset/drivetrack`, each one with the structure described above.

  Args:
    npz_filenames: List of npz file names to compute metrics over.
                   Each filename must exist in both `ground_truth_dir` and
                   `predictions_dir`.
    ground_truth_dir: Path to the TAP-3D dataset, format described above.
    predictions_dir: Path to the TAP-3D predictions, format described above.
    depth_scalings: List of strings, describing which depth scaling strategies
                   to use. See `compute_tapvid3d_metrics()` in metrics.py for
                   available scaling options and their descriptions.
    metric_eval_resolution: The resolution at which to evaluate the metrics,
                            by default is 256, which is used for all results
                            in the TAPVid-3D paper and original TAPVid paper.

  Returns:
    A dictionary of dictionaries: the outer dict mapping different
    depth scaling strategies to an inner metrics dict (mapping a metric name
    to the corresponding metric score).
  """
  metrics_all_videos = []
  for npy_file in tqdm.tqdm(npz_filenames, total=len(npz_filenames)):
    gt_file = os.path.join(ground_truth_dir, npy_file)
    with open(gt_file, 'rb') as in_f:
      in_npz = np.load(in_f, allow_pickle=True)
      images_jpeg_bytes = in_npz['images_jpeg_bytes']
      queries_xyt = in_npz['queries_xyt']
      tracks_xyz = in_npz['tracks_XYZ']
      visibles = in_npz['visibility']
      intrinsics_params = in_npz['fx_fy_cx_cy']

    # Simple rescaling from original video resolution to the resolution
    # at which we want to compute the metrics (metric_eval_resolution)
    # Since camera intrinsics are for original video resolution, we need to
    # rescale them as well.
    # `intrinsics_params` is in format [fx, fy, cx, cy].
    video_height, video_width = get_jpeg_byte_hw(images_jpeg_bytes[0])
    smallest_side_length = metric_eval_resolution
    (_, _), scaling_factor = (
        get_new_hw_with_given_smallest_side_length(
            orig_height=video_height,
            orig_width=video_width,
            smallest_side_length=smallest_side_length,
        )
    )
    intrinsics_params_resized = intrinsics_params * scaling_factor

    prediction_file = os.path.join(predictions_dir, npy_file)
    try:
      with open(prediction_file, 'rb') as in_f:
        predictor_data = np.load(in_f, allow_pickle=True)
        predicted_tracks_xyz = predictor_data['tracks_XYZ']
        predicted_visibility = predictor_data['visibility']

    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Failed to read %s', prediction_file)
      failure_metrics_dict = {
          scaling: ZERO_METRICS_DICT for scaling in depth_scalings
      }
      metrics_all_videos.append(failure_metrics_dict)
      if _DEBUG.value:
        logging.info('Stopping after one video, debug run.')
        break
      continue

    video_metrics = {}
    for depth_scaling in depth_scalings:
      try:
        metrics_for_scale = metrics.compute_tapvid3d_metrics(
            gt_occluded=np.logical_not(visibles),
            gt_tracks=tracks_xyz,
            pred_occluded=np.logical_not(predicted_visibility),
            pred_tracks=predicted_tracks_xyz,
            intrinsics_params=intrinsics_params_resized,
            scaling=depth_scaling,
            query_points=queries_xyt[..., ::-1],
            order='t n',
        )
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception('Failed to compute metrics for %s', npy_file)
        metrics_for_scale = ZERO_METRICS_DICT
      video_metrics[depth_scaling] = metrics_for_scale
    metrics_all_videos.append(video_metrics)

    if _DEBUG.value:
      logging.info('Stopping after one video, debug run.')
      break

  avg_metrics = get_average_over_metrics(metrics_all_videos)
  return avg_metrics


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  metrics_all_sources = []
  for data_source in _DATA_SOURCES_TO_EVALUATE.value:
    if _USE_MINIVAL.value:
      all_npz_files = tapvid3d_splits.get_minival_files(subset=data_source)
    else:
      all_npz_files = tapvid3d_splits.get_full_eval_files(subset=data_source)
    source_gt_dir = os.path.join(_TAPVID3D_DIR.value, data_source)
    source_pred_dir = os.path.join(_TAPVID3D_PREDICTIONS.value, data_source)
    source_metrics = evaluate_data_source(
        npz_filenames=all_npz_files,
        ground_truth_dir=source_gt_dir,
        predictions_dir=source_pred_dir,
        depth_scalings=_DEPTH_SCALINGS.value,
    )
    metrics_all_sources.append(source_metrics)
    logging.info('Metrics for data source %s', data_source)
    logging.info(source_metrics)

  avg_metrics = get_average_over_metrics(metrics_all_sources)
  logging.info('Metrics, averaged across all data sources:')
  logging.info(avg_metrics)

  logging.info('Finished computing metrics!')


if __name__ == '__main__':
  app.run(main)

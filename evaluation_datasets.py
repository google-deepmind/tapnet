# Copyright 2023 DeepMind Technologies Limited
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

"""Evaluation dataset creation functions."""

import csv
import functools
import io
import os
from os import path
import pickle
import random
from typing import Iterable, Mapping, Optional, Tuple, Union

from absl import logging

from kubric.challenges.point_tracking import dataset
import mediapy as media
import numpy as np
from PIL import Image
import scipy.io as sio
import tensorflow as tf
import tensorflow_datasets as tfds

from tapnet.utils import transforms

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  """Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
  return media.resize_video(video, output_size[1:3])


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts.

  Within Thresh, Occ.

  Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=(1, 2),
  ) / np.sum(evaluation_points)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=(1, 2),
    )
    count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=(1, 2)
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics


def latex_table(mean_scalars: Mapping[str, float]) -> str:
  """Generate a latex table for displaying TAP-Vid and PCK metrics."""
  if 'average_jaccard' in mean_scalars:
    latex_fields = [
        'average_jaccard',
        'average_pts_within_thresh',
        'occlusion_accuracy',
        'jaccard_1',
        'jaccard_2',
        'jaccard_4',
        'jaccard_8',
        'jaccard_16',
        'pts_within_1',
        'pts_within_2',
        'pts_within_4',
        'pts_within_8',
        'pts_within_16',
    ]
    header = (
        'AJ & $<\\delta^{x}_{avg}$ & OA & Jac. $\\delta^{0}$ & '
        + 'Jac. $\\delta^{1}$ & Jac. $\\delta^{2}$ & '
        + 'Jac. $\\delta^{3}$ & Jac. $\\delta^{4}$ & $<\\delta^{0}$ & '
        + '$<\\delta^{1}$ & $<\\delta^{2}$ & $<\\delta^{3}$ & '
        + '$<\\delta^{4}$'
    )
  else:
    latex_fields = ['PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5']
    header = ' & '.join(latex_fields)

  body = ' & '.join(
      [f'{float(np.array(mean_scalars[x]*100)):.3}' for x in latex_fields]
  )
  return '\n'.join([header, body])


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, sample queries
  strided every query_stride frames, ignoring points that are not visible
  at the selected frames.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.
    query_stride: When sampling query points, search for un-occluded points
      every query_stride frames and convert each one into a query.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3].  The video
        has floats scaled to the range [-1, 1].
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1].
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1].
      trackgroup: Index of the original track that each query point was
        sampled from.  This is useful for visualization.
  """
  tracks = []
  occs = []
  queries = []
  trackgroups = []
  total = 0
  trackgroup = np.arange(target_occluded.shape[0])
  for i in range(0, target_occluded.shape[1], query_stride):
    mask = target_occluded[:, i] == 0
    query = np.stack(
        [
            i * np.ones(target_occluded.shape[0:1]),
            target_points[:, i, 1],
            target_points[:, i, 0],
        ],
        axis=-1,
    )
    queries.append(query[mask])
    tracks.append(target_points[mask])
    occs.append(target_occluded[mask])
    trackgroups.append(trackgroup[mask])
    total += np.array(np.sum(target_occluded[:, i] == 0))

  return {
      'video': frames[np.newaxis, ...],
      'query_points': np.concatenate(queries, axis=0)[np.newaxis, ...],
      'target_points': np.concatenate(tracks, axis=0)[np.newaxis, ...],
      'occluded': np.concatenate(occs, axis=0)[np.newaxis, ...],
      'trackgroup': np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
  }


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, use the first
  visible point in each track as the query.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3]
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1]
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1]
  """

  valid = np.sum(~target_occluded, axis=1) > 0
  target_points = target_points[valid, :]
  target_occluded = target_occluded[valid, :]

  query_points = []
  for i in range(target_points.shape[0]):
    index = np.where(target_occluded[i] == 0)[0][0]
    x, y = target_points[i, index, 0], target_points[i, index, 1]
    query_points.append(np.array([index, y, x]))  # [t, y, x]
  query_points = np.stack(query_points, axis=0)

  return {
      'video': frames[np.newaxis, ...],
      'query_points': query_points[np.newaxis, ...],
      'target_points': target_points[np.newaxis, ...],
      'occluded': target_occluded[np.newaxis, ...],
  }


def create_jhmdb_dataset(
    jhmdb_path: str, resolution: Optional[Tuple[int, int]] = (256, 256)
) -> Iterable[DatasetElement]:
  """JHMDB dataset, including fields required for PCK evaluation."""
  videos = []
  for file in tf.io.gfile.listdir(path.join(gt_dir, 'splits')):
    # JHMDB file containing the first split, which is standard for this type of
    # evaluation.
    if not file.endswith('split1.txt'):
      continue

    video_folder = '_'.join(file.split('_')[:-2])
    for video in tf.io.gfile.GFile(path.join(gt_dir, 'splits', file), 'r'):
      video, traintest = video.split()
      video, _ = video.split('.')

      traintest = int(traintest)
      video_path = path.join(video_folder, video)

      if traintest == 2:
        videos.append(video_path)

  if not videos:
    raise ValueError('No JHMDB videos found in directory ' + str(jhmdb_path))

  # Shuffle so numbers converge faster.
  random.shuffle(videos)

  for video in videos:
    logging.info(video)
    joints = path.join(gt_dir, 'joint_positions', video, 'joint_positions.mat')

    if not tf.io.gfile.exists(joints):
      logging.info('skip %s', video)
      continue

    gt_pose = sio.loadmat(tf.io.gfile.GFile(joints, 'rb'))['pos_img']
    gt_pose = np.transpose(gt_pose, [1, 2, 0])
    frames = path.join(gt_dir, 'Rename_Images', video, '*.png')
    framefil = tf.io.gfile.glob(frames)
    framefil.sort()

    def read_frame(f):
      im = Image.open(tf.io.gfile.GFile(f, 'rb'))
      im = im.convert('RGB')
      im_data = np.array(im.getdata(), np.uint8)
      return im_data.reshape([im.size[1], im.size[0], 3])

    frames = [read_frame(x) for x in framefil]
    frames = np.stack(frames)
    height = frames.shape[1]
    width = frames.shape[2]
    invalid_x = np.logical_or(
        gt_pose[:, 0:1, 0] < 0,
        gt_pose[:, 0:1, 0] >= width,
    )
    invalid_y = np.logical_or(
        gt_pose[:, 0:1, 1] < 0,
        gt_pose[:, 0:1, 1] >= height,
    )
    invalid = np.logical_or(invalid_x, invalid_y)
    invalid = np.tile(invalid, [1, gt_pose.shape[1]])
    invalid = invalid[:, :, np.newaxis].astype(np.float32)
    gt_pose_orig = gt_pose

    if resolution is not None and resolution != frames.shape[1:3]:
      frames = resize_video(frames, resolution)
    frames = frames / (255.0 / 2.0) - 1.0
    queries = gt_pose[:, 0]
    queries = np.concatenate(
        [queries[..., 0:1] * 0, queries[..., ::-1]],
        axis=-1,
    )
    gt_pose = transforms.convert_grid_coordinates(
        gt_pose,
        np.array([width, height]),
        np.array([frames.shape[2], frames.shape[1]]),
    )
    # Set invalid poses to -1 (outside the frame)
    gt_pose = (1.0 - invalid) * gt_pose + invalid * (-1.0)

    if gt_pose.shape[1] < frames.shape[0]:
      # Some videos have pose sequences that are shorter than the frame
      # sequence (usually because the person disappears).  In this case,
      # truncate the video.
      logging.warning('short video!!')
      frames = frames[: gt_pose.shape[1]]

    converted = {
        'video': frames[np.newaxis, ...],
        'query_points': queries[np.newaxis, ...],
        'target_points': gt_pose[np.newaxis, ...],
        'gt_pose': gt_pose[np.newaxis, ...],
        'gt_pose_orig': gt_pose_orig[np.newaxis, ...],
        'occluded': gt_pose[np.newaxis, ..., 0] * 0,
        'fname': video,
        'im_size': np.array([height, width]),
    }
    yield {'jhmdb': converted}


def create_kubric_eval_train_dataset(
    mode: str,
    train_size: Tuple[int, int] = (256, 256),
    max_dataset_size: int = 100,
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on Kubric training data."""
  res = dataset.create_point_tracking_dataset(
      split='train',
      train_size=train_size,
      batch_dims=[1],
      shuffle_buffer_size=None,
      repeat=False,
      vflip='vflip' in mode,
      random_crop=False,
  )
  np_ds = tfds.as_numpy(res)

  num_returned = 0
  for data in np_ds:
    if num_returned >= max_dataset_size:
      break
    num_returned += 1
    yield {'kubric': data}


def create_kubric_eval_dataset(
    mode: str, train_size: Tuple[int, int] = (256, 256)
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on Kubric val data."""
  res = dataset.create_point_tracking_dataset(
      split='validation',
      train_size=train_size,
      batch_dims=[1],
      shuffle_buffer_size=None,
      repeat=False,
      vflip='vflip' in mode,
      random_crop=False,
  )
  np_ds = tfds.as_numpy(res)

  for data in np_ds:
    yield {'kubric': data}


def create_davis_dataset(
    davis_points_path: str,
    query_mode: str = 'strided',
    full_resolution=False,
    resolution: Optional[Tuple[int, int]] = (256, 256),
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on DAVIS data."""
  pickle_path = davis_points_path

  with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    davis_points_dataset = pickle.load(f)

  if full_resolution:
    ds, _ = tfds.load(
        'davis/full_resolution', split='validation', with_info=True
    )
    to_iterate = tfds.as_numpy(ds)
  else:
    to_iterate = davis_points_dataset.keys()

  for tmp in to_iterate:
    if full_resolution:
      frames = tmp['video']['frames']
      video_name = tmp['metadata']['video_name'].decode()
    else:
      video_name = tmp
      frames = davis_points_dataset[video_name]['video']
      if resolution is not None and resolution != frames.shape[1:3]:
        frames = resize_video(frames, resolution)

    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
    target_points = davis_points_dataset[video_name]['points']
    target_occ = davis_points_dataset[video_name]['occluded']
    target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

    if query_mode == 'strided':
      converted = sample_queries_strided(target_occ, target_points, frames)
    elif query_mode == 'first':
      converted = sample_queries_first(target_occ, target_points, frames)
    else:
      raise ValueError(f'Unknown query mode {query_mode}.')

    yield {'davis': converted}


def create_rgb_stacking_dataset(
    robotics_points_path: str,
    query_mode: str = 'strided',
    resolution: Optional[Tuple[int, int]] = (256, 256),
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on robotics data."""
  pickle_path = robotics_points_path

  with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    robotics_points_dataset = pickle.load(f)

  for example in robotics_points_dataset:
    frames = example['video']
    if resolution is not None and resolution != frames.shape[1:3]:
      frames = resize_video(frames, resolution)
    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
    target_points = example['points']
    target_occ = example['occluded']
    target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

    if query_mode == 'strided':
      converted = sample_queries_strided(target_occ, target_points, frames)
    elif query_mode == 'first':
      converted = sample_queries_first(target_occ, target_points, frames)
    else:
      raise ValueError(f'Unknown query mode {query_mode}.')

    yield {'robotics': converted}


def create_kinetics_dataset(
    kinetics_path: str, query_mode: str = 'strided',
    resolution: Optional[Tuple[int, int]] = (256, 256),
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on Kinetics point tracking."""

  all_paths = tf.io.gfile.glob(path.join(kinetics_path, '*_of_0010.pkl'))
  for pickle_path in all_paths:
    with open(pickle_path, 'rb') as f:
      data = pickle.load(f)
      if isinstance(data, dict):
        data = list(data.values())

    # idx = random.randint(0, len(data) - 1)
    for idx in range(len(data)):
      example = data[idx]

      frames = example['video']

      if isinstance(frames[0], bytes):
        # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
        def decode(frame):
          byteio = io.BytesIO(frame)
          img = Image.open(byteio)
          return np.array(img)

        frames = np.array([decode(frame) for frame in frames])

      if resolution is not None and resolution != frames.shape[1:3]:
        frames = resize_video(frames, resolution)

      frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
      target_points = example['points']
      target_occ = example['occluded']
      target_points *= np.array([frames.shape[2], frames.shape[1]])

      if query_mode == 'strided':
        converted = sample_queries_strided(target_occ, target_points, frames)
      elif query_mode == 'first':
        converted = sample_queries_first(target_occ, target_points, frames)
      else:
        raise ValueError(f'Unknown query mode {query_mode}.')

      yield {'kinetics': converted}


def create_csv_dataset(
    dataset_name: str,
    csv_path: str,
    video_base_path: str,
    query_mode: str = 'strided',
    resolution: Optional[Tuple[int, int]] = (256, 256),
    max_video_frames: Optional[int] = 1000,
) -> Iterable[DatasetElement]:
  """Create an evaluation iterator out of human annotations and videos.

  Args:
    dataset_name: Name to the dataset.
    csv_path: Path to annotations csv.
    video_base_path: Path to annotated videos.
    query_mode: sample query points from first frame or strided.
    resolution: The video resolution in (height, width).
    max_video_frames: Max length of annotated video.

  Yields:
    Samples for evaluation.
  """
  point_tracks_all = dict()
  with tf.io.gfile.GFile(csv_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      video_id = row[0]
      point_tracks = np.array(row[1:]).reshape(-1, 3)
      if video_id in point_tracks_all:
        point_tracks_all[video_id].append(point_tracks)
      else:
        point_tracks_all[video_id] = [point_tracks]

  for video_id in point_tracks_all:
    if video_id.endswith('.mp4'):
      video_path = path.join(video_base_path, video_id)
    else:
      video_path = path.join(video_base_path, video_id + '.mp4')
    frames = media.read_video(video_path)
    if resolution is not None and resolution != frames.shape[1:3]:
      frames = media.resize_video(frames, resolution)
    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0

    point_tracks = np.stack(point_tracks_all[video_id], axis=0)
    point_tracks = point_tracks.astype(np.float32)
    if frames.shape[0] < point_tracks.shape[1]:
      logging.info('Warning: short video!')
      point_tracks = point_tracks[:, : frames.shape[0]]
    point_tracks, occluded = point_tracks[..., 0:2], point_tracks[..., 2]
    occluded = occluded > 0
    target_points = point_tracks * np.array([frames.shape[2], frames.shape[1]])

    num_splits = int(np.ceil(frames.shape[0] / max_video_frames))
    if num_splits > 1:
      print(f'Going to split the video {video_id} into {num_splits}')
    for i in range(num_splits):
      start_index = i * frames.shape[0] // num_splits
      end_index = (i + 1) * frames.shape[0] // num_splits
      sub_occluded = occluded[:, start_index:end_index]
      sub_target_points = target_points[:, start_index:end_index]
      sub_frames = frames[start_index:end_index]

      if query_mode == 'strided':
        converted = sample_queries_strided(
            sub_occluded, sub_target_points, sub_frames
        )
      elif query_mode == 'first':
        converted = sample_queries_first(
            sub_occluded, sub_target_points, sub_frames
        )
      else:
        raise ValueError(f'Unknown query mode {query_mode}.')

      yield {dataset_name: converted}

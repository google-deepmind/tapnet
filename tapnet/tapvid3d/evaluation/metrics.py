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

"""TAPVid-3D metrics."""

from typing import Mapping, Optional
import einops
import numpy as np


def get_pointwise_threshold_multiplier(
    gt_tracks: np.ndarray, intrinsics_params: np.ndarray
) -> np.ndarray | float:
  mean_focal_length = np.sqrt(
      intrinsics_params[..., 0] * intrinsics_params[..., 1]
  )
  return gt_tracks[..., -1] / mean_focal_length[..., np.newaxis, np.newaxis]


PIXEL_TO_FIXED_METRIC_THRESH = {
    1: 0.01,
    2: 0.04,
    4: 0.16,
    8: 0.64,
    16: 2.56,
}


def create_local_tracks(
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Gather all points within a threshold distance of ground truth."""
  out_gt_tr = []
  out_gt_occ = []
  out_pr_tr = []
  out_pr_occ = []

  # for each track, find the points that are within a threshold distance of
  # the track's point on the corresponding frame and grab them
  for idx in range(gt_occluded.shape[0]):
    diffs = gt_tracks - gt_tracks[idx : idx + 1]
    is_neighbor = np.sum(np.square(diffs), axis=-1) < thresh * thresh
    is_neighbor = np.reshape(is_neighbor, [-1])

    def grab(x):
      x = np.reshape(x, [-1, x.shape[-1]])
      return x[is_neighbor]  # pylint: disable=cell-var-from-loop

    out_gt_tr.append(grab(gt_tracks))
    out_pr_tr.append(grab(pred_tracks))
    out_gt_occ.append(grab(gt_occluded[..., np.newaxis]))
    out_pr_occ.append(grab(pred_occluded[..., np.newaxis]))

  # need to pad to the longest length before stacking
  largest = np.max([x.shape[0] for x in out_gt_tr])

  def pad(x):
    res = np.zeros([largest, x.shape[-1]], dtype=x.dtype)
    res[: x.shape[0]] = x
    return res

  out_gt_tr = np.stack([pad(x) for x in out_gt_tr])
  out_pr_tr = np.stack([pad(x) for x in out_pr_tr])
  valid = np.stack([pad(np.ones_like(x)) for x in out_gt_occ])[..., 0]
  out_gt_occ = np.stack([pad(x) for x in out_gt_occ])[..., 0]
  out_pr_occ = np.stack([pad(x) for x in out_pr_occ])[..., 0]
  weighting = np.sum((1.0 - gt_occluded), axis=1, keepdims=True) / np.maximum(
      1.0, np.sum((1.0 - out_gt_occ) * valid, axis=1, keepdims=True)
  )

  return out_gt_occ, out_gt_tr, out_pr_occ, out_pr_tr, valid * weighting


def compute_tapvid3d_metrics(
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    intrinsics_params: np.ndarray,
    get_trackwise_metrics: bool = False,
    scaling: str = 'median',
    query_points: Optional[np.ndarray] = None,
    use_fixed_metric_threshold: bool = False,
    local_neighborhood_thresh: Optional[float] = 0.05,
    order: str = 'n t',
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.).

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
     gt_occluded: A boolean array, generally of shape [b, n, t] or [n, t], where
       t is the number of frames and n is the number of tracks. True indicates
       that the point is occluded. Must be consistent with 'order' parameter, so
       if passing in [t, n] or [b, t, n] instead, order string must reflect
       this!
     gt_tracks: The target points, of shape [b, n, t, 3] or [n, t, 3], unless
       specified otherwise in the order parameter. Each point is in the format
       [x, y, z].
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     intrinsics_params: camera intrinsic parameters, [fx, fy, cx, cy].  Full
       intrinsic matrix has the form [[fx, 0, cx],[0, fy, cy],[0,0,1]]
     get_trackwise_metrics: if True, the metrics will be computed for every
       track (rather than every video, which is the default).  This means every
       output tensor will have an extra axis [batch, num_tracks] rather than
       simply (batch).
     scaling: How to rescale the estimated point tracks to match the global
       scale of the ground truth points.  Possible options are "median" (to
       scale by the median norm of points visible in both prediction and ground
       truth; default), "mean" (same as "median", but using the Euclidean mean),
       "per_trajectory" which rescales predicted the predicted track so that the
       predicted depth on the query frame matches the ground truth,
       "local_neighborhood" which gathers, for every track, all points that are
       within a threshold (local_neighborhood_thresh) and treats all such points
       as a single track, afterward performing "per_trajectory" scaling, "none"
       (don't rescale points at all), and "reproduce_2d" which scales every
       point to match ground truth depth without changing the reprojection,
       which will reproduce the thresholds from 2D TAP.  Note that this won't
       exactly match the output of compute_tapvid_metrics because that function
       ignores the query points.
     query_points: query points, of shape [b, n, 3] or [n, 3] t/y/x points. Only
       needed if scaling == "per_trajectory", so we know which frame to use for
       rescaling.
     use_fixed_metric_threshold: if True, the metrics will be computed using
       fixed metric bubble thresholds, rather than the depth-adaptive thresholds
       scaled depth and by the camera intrinsics.
     local_neighborhood_thresh: distance threshold for local_neighborhood
       scaling.
      order: order of the prediction and visibility tensors.  Can be 'n t'
        (default), 't n', or 'b n t' or 'b t n'.

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given back-projected pixel threshold,
        ignoring occlusion prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  # Adjust variable input shapes and orders to expected 'b n t',
  # except in case of local_neighborhood, which expects 'n t'.
  batched_input = len(order.split(' ')) == 3
  if scaling == 'local_neighborhood':
    assert (
        not batched_input
    ), "Local neighborhood doesn't support batched inputs."
    output_order = 'n t'
  else:
    if batched_input:
      output_order = 'b n t'
    else:
      output_order = '() n t'  # Append batch axis.

  gt_occluded = einops.rearrange(gt_occluded, f'{order} -> {output_order}')
  pred_occluded = einops.rearrange(pred_occluded, f'{order} -> {output_order}')
  gt_tracks = einops.rearrange(gt_tracks, f'{order} d -> {output_order} d')
  pred_tracks = einops.rearrange(pred_tracks, f'{order} d -> {output_order} d')

  summing_axis = (-1,) if get_trackwise_metrics else (-2, -1)
  evaluation_weights = np.ones(gt_occluded.shape)

  metrics = {}

  pred_norms = np.sqrt(
      np.maximum(1e-12, np.sum(np.square(pred_tracks), axis=-1))
  )
  gt_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(gt_tracks), axis=-1)))
  if scaling == 'reproduce_2d':
    scale_factor = gt_tracks[..., -1:] / pred_tracks[..., -1:]
  elif scaling == 'per_trajectory' or scaling == 'local_neighborhood':
    query_frame = np.round(query_points[..., 0]).astype(np.int32)[
        ..., np.newaxis
    ]

    def do_take(x):
      took = np.take_along_axis(x, query_frame, axis=-1)
      return np.maximum(took, 1e-12)[..., np.newaxis]

    scale_factor = do_take(gt_tracks[..., -1]) / do_take(pred_tracks[..., -1])
    if scaling == 'local_neighborhood':
      gt_occluded, gt_tracks, pred_occluded, pred_tracks, evaluation_weights = (
          create_local_tracks(
              gt_occluded,
              gt_tracks,
              pred_occluded,
              pred_tracks,
              thresh=local_neighborhood_thresh,
          )
      )
  else:
    either_occluded = np.logical_or(gt_occluded, pred_occluded)
    nan_mat = np.full(pred_norms.shape, np.nan)
    pred_norms = np.where(either_occluded, nan_mat, pred_norms)
    gt_norms = np.where(either_occluded, nan_mat, gt_norms)
    if scaling == 'median':
      scale_factor = np.nanmedian(
          gt_norms, axis=(-2, -1), keepdims=True
      ) / np.nanmedian(pred_norms, axis=(-2, -1), keepdims=True)
    elif scaling == 'mean':
      scale_factor = np.nanmean(
          gt_norms, axis=(-2, -1), keepdims=True
      ) / np.nanmean(pred_norms, axis=(-2, -1), keepdims=True)
    elif scaling == 'none':
      scale_factor = 1.0
    elif scaling == 'median_on_queries':
      range_n_pts = np.arange(pred_norms.shape[-2])
      query_frame = np.round(query_points[..., 0]).astype(np.int32).squeeze()
      pred_norms = pred_norms[:, range_n_pts, query_frame][..., None]
      gt_norms = gt_norms[:, range_n_pts, query_frame][..., None]
      scale_factor = np.nanmedian(
          gt_norms, axis=(-2, -1), keepdims=True
      ) / np.nanmedian(pred_norms, axis=(-2, -1), keepdims=True)
    else:
      raise ValueError('Unknown scaling:' + scaling)

  pred_tracks = pred_tracks * scale_factor

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  metrics['occlusion_accuracy'] = np.sum(
      np.equal(pred_occluded, gt_occluded) * evaluation_weights,
      axis=summing_axis,
  ) / np.sum(evaluation_weights, axis=summing_axis)

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    if use_fixed_metric_threshold:
      pointwise_thresh = PIXEL_TO_FIXED_METRIC_THRESH[thresh]
    else:
      multiplier = get_pointwise_threshold_multiplier(
          gt_tracks, intrinsics_params
      )
      pointwise_thresh = thresh * multiplier
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(pointwise_thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct * evaluation_weights,
        axis=summing_axis,
    )
    count_visible_points = np.sum(
        visible * evaluation_weights,
        axis=summing_axis
    )
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        (is_correct & pred_visible) * evaluation_weights,
        axis=summing_axis)

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible * evaluation_weights, axis=summing_axis)
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(
        false_positives * evaluation_weights, axis=summing_axis
    )
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)

  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=-2),
      axis=-2,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=-2),
      axis=-2,
  )

  return metrics

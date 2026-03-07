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

"""Functions for computing the AJ_RD metric."""

from typing import Any
import numpy as np
import torch


def calculate_jaccard_for_segment(
    pred_track_segment: torch.Tensor,
    pred_vis_segment: torch.Tensor,
    gt_track_segment: torch.Tensor,
    gt_vis_segment: torch.Tensor,
    dist_threshold: float,
) -> torch.Tensor:
  """Computes Jaccard metric for a single track segment for given threshold.

  Assumes inputs are torch tensors of shape [L, ...] where L is segment length.

  Args:
    pred_track_segment: Predicted track segment of shape [L, 2].
    pred_vis_segment: Predicted visibility segment of shape [L].
    gt_track_segment: Ground truth track segment of shape [L, 2].
    gt_vis_segment: Ground truth visibility segment of shape [L].
    dist_threshold: The distance threshold.

  Returns:
    The Jaccard similarity for the segment as a float tensor, or NaN if the
    denominator is zero.
  """
  within_dist = (
      torch.linalg.norm(pred_track_segment - gt_track_segment, dim=-1)
      <= dist_threshold
  )
  is_correct = within_dist & gt_vis_segment

  true_positives = torch.sum(is_correct & pred_vis_segment)

  gt_positives = torch.sum(gt_vis_segment)
  false_positives = (~gt_vis_segment) & pred_vis_segment
  false_positives = false_positives | ((~within_dist) & pred_vis_segment)
  false_positives = torch.sum(false_positives)

  denominator = gt_positives + false_positives
  if denominator == 0:
    return torch.tensor(float('nan'), device=pred_track_segment.device)
  return true_positives.float() / denominator


def count_consecutive_invisibility(is_visible: torch.Tensor) -> torch.Tensor:
  """Counts consecutive invisible frames ending at the previous timestep.

  Args:
      is_visible: boolean tensor of shape B,T,N indicating point visibility.

  Returns:
      d: integer tensor of shape B,T,N where d[b,t,n] contains the number
         of consecutive frames up to t-1 for which the point (b,n) was
         not visible. If is_visible[b,t-1,n] is True, d[b,t,n] is 0.
         If is_visible[b,t-1,n]...is_visible[b,t-k,n] are False, and
         is_visible[b,t-k-1,n] is True (or t-k=0), then d[b,t,n]=k.
  """
  batch_size, num_frames, num_points = is_visible.shape
  d = torch.zeros(
      (batch_size, num_frames, num_points),
      dtype=torch.int,
      device=is_visible.device,
  )
  for t in range(1, num_frames):
    d[:, t, :] = torch.where(~is_visible[:, t - 1, :], d[:, t - 1, :] + 1, 0)
  return d


def compute_raw_redetection_stats(
    pred_tracks: torch.Tensor,  # B,T,N,2
    pred_visible: torch.Tensor,  # B,T,N
    gt_tracks: torch.Tensor,  # B,T,N,2
    gt_visible: torch.Tensor,  # B,T,N
    dist_thresholds: list[int] | None = None,
) -> dict[str, Any] | None:
  """Computes raw redetection statistics for every eligible reappearance event.

  These stats can be processed later to compute metrics.

  Args:
      pred_tracks: Predicted point tracks (B,T,N,2).
      pred_visible: Predicted point visibility (B,T,N).
      gt_tracks: Ground truth point tracks (B,T,N,2).
      gt_visible: Ground truth point visibility (B,T,N).
      dist_thresholds: List of pixel distance thresholds for successful
        detection.

  Returns:
      A dictionary containing:
          - 'indices': Indices (b, t_r, n) of eligible reappearance events.
          - 'durations': The invisibility duration for each eligible event.
          - 'metrics_per_dist': A dictionary where keys are distance thresholds
          D,
            and values are dictionaries containing 'aj_rd', a tensor of AJ_RD
            scores for each eligible event at that distance threshold.
      Returns None if no eligible reappearance events are found.
  """
  if dist_thresholds is None:
    dist_thresholds = [1, 2, 4, 8, 16]

  _, num_frames, _, _ = pred_tracks.shape
  gt_visible = gt_visible.bool()
  pred_visible = pred_visible.bool()

  reapp_mask = torch.zeros_like(gt_visible)
  reapp_mask[:, 1:, :] = gt_visible[:, 1:, :] & ~gt_visible[:, :-1, :]

  d_tensor = count_consecutive_invisibility(
      gt_visible
  )  # d_tensor[b,t,n] = d if t is t_r

  reapp_indices = torch.where(reapp_mask)
  num_reapp_events = reapp_indices[0].shape[0]

  if num_reapp_events == 0:
    return None

  durations = d_tensor[reapp_mask]

  # Filter for eligible events: only include event i if d_i > max d_j
  # for j<i in same track
  is_eligible_event = torch.zeros(
      num_reapp_events, dtype=torch.bool, device=pred_tracks.device
  )
  unique_tracks = {}  # map (b,n) to list of (t_r, duration, event_idx)
  for i in range(num_reapp_events):
    b, t_r, n = (
        reapp_indices[0][i].item(),
        reapp_indices[1][i].item(),
        reapp_indices[2][i].item(),
    )
    if (b, n) not in unique_tracks:
      unique_tracks[(b, n)] = []
    unique_tracks[(b, n)].append((t_r, durations[i].item(), i))

  for track_id in unique_tracks:
    events = sorted(unique_tracks[track_id], key=lambda x: x[0])  # sort by t_r
    max_d_seen = -1
    for _, d, event_idx in events:
      if d > max_d_seen:
        is_eligible_event[event_idx] = True
        max_d_seen = d

  if not torch.any(is_eligible_event):
    return None

  # Select only eligible events for metrics
  eligible_reapp_indices = tuple(t[is_eligible_event] for t in reapp_indices)
  eligible_durations = durations[is_eligible_event]
  num_eligible_events = eligible_reapp_indices[0].shape[0]

  results = {
      'indices': eligible_reapp_indices,
      'durations': eligible_durations,
      'metrics_per_dist': {},
  }

  for d_thresh in dist_thresholds:
    aj_rd_d = torch.full(
        (num_eligible_events,),
        float('nan'),
        dtype=torch.float,
        device=pred_tracks.device,
    )

    for i in range(num_eligible_events):
      b, t_r, n = (
          eligible_reapp_indices[0][i],
          eligible_reapp_indices[1][i],
          eligible_reapp_indices[2][i],
      )

      # AJ_RD: From t_r to end of sequence
      t_end = num_frames
      if t_end > t_r:
        aj_rd_d[i] = calculate_jaccard_for_segment(
            pred_tracks[b, t_r:t_end, n],
            pred_visible[b, t_r:t_end, n],
            gt_tracks[b, t_r:t_end, n],
            gt_visible[b, t_r:t_end, n],
            d_thresh,
        )

    results['metrics_per_dist'][d_thresh] = {
        'aj_rd': aj_rd_d,
    }
  return results


def compute_redetection_metrics(
    pred_tracks: torch.Tensor,  # B,T,N,2
    pred_visible: torch.Tensor,  # B,T,N
    gt_tracks: torch.Tensor,  # B,T,N,2
    gt_visible: torch.Tensor,  # B,T,N
    dist_thresholds: list[int] | None = None,
    d_min_thresholds: list[int] | None = None,
) -> dict[str, float]:
  """Computes redetection metrics AJ_RD based on provided predictions.

  Metrics are computed cumulatively for reappearance events with invisibility
  duration >= d_min, for each d_min in d_min_thresholds.

  Args:
      pred_tracks: Predicted point tracks (B,T,N,2).
      pred_visible: Predicted point visibility (B,T,N).
      gt_tracks: Ground truth point tracks (B,T,N,2).
      gt_visible: Ground truth point visibility (B,T,N).
      dist_thresholds: List of pixel distance thresholds (D) for successful
        detection.
      d_min_thresholds: List of minimum duration thresholds d' for computing
        Metric(d_min=d').

  Returns:
      A dictionary containing AJ_RD metrics for each combination of
      threshold D, and d_min threshold.
  """
  if dist_thresholds is None:
    dist_thresholds = [1, 2, 4, 8, 16]
  if d_min_thresholds is None:
    d_min_thresholds = [1, 4, 16, 64, 256]

  raw_stats = compute_raw_redetection_stats(
      pred_tracks,
      pred_visible,
      gt_tracks,
      gt_visible,
      dist_thresholds,
  )

  metrics = {}
  if raw_stats is None:
    for d_min in d_min_thresholds:
      for d_thresh in dist_thresholds:
        metrics[f'AJ_RD_D{d_thresh}_dmin{d_min}'] = float('nan')
      metrics[f'AJ_RD_dmin{d_min}'] = float('nan')
    metrics['AJ_RD'] = float('nan')
    return metrics

  durations = raw_stats['durations']

  for d_min in d_min_thresholds:
    d_min_mask = durations >= d_min
    num_d_min_events = torch.sum(d_min_mask).item()

    if num_d_min_events == 0:
      for d_thresh in dist_thresholds:
        metrics[f'AJ_RD_D{d_thresh}_dmin{d_min}'] = float('nan')
      metrics[f'AJ_RD_dmin{d_min}'] = float('nan')
      continue

    for d_thresh in dist_thresholds:
      stats_d = raw_stats['metrics_per_dist'][d_thresh]

      # AJ_RD
      aj_rd_d = stats_d['aj_rd']
      d_min_aj_rd_d = aj_rd_d[d_min_mask]
      d_min_aj_rd_valid = d_min_aj_rd_d[~torch.isnan(d_min_aj_rd_d)]
      if len(d_min_aj_rd_valid) > 0:
        metrics[f'AJ_RD_D{d_thresh}_dmin{d_min}'] = torch.mean(
            d_min_aj_rd_valid
        ).item()
      else:
        metrics[f'AJ_RD_D{d_thresh}_dmin{d_min}'] = float('nan')

    aj_rd_values_for_dmin = [
        metrics[f'AJ_RD_D{d_thresh}_dmin{d_min}']
        for d_thresh in dist_thresholds
    ]
    metrics[f'AJ_RD_dmin{d_min}'] = (
        np.nanmean(aj_rd_values_for_dmin)
        if any(~np.isnan(v) for v in aj_rd_values_for_dmin)
        else float('nan')
    )

  # Add raw stats to metrics for debugging
  if raw_stats is not None:
    metrics = {**metrics, **{f'raw_stats/{k}': v for k, v in raw_stats.items()}}

  # Compute final AJ_RD score
  final_aj_rd_vals = []
  for d in d_min_thresholds:
    if f'AJ_RD_dmin{d}' in metrics:
      final_aj_rd_vals.append(metrics[f'AJ_RD_dmin{d}'])
  metrics['AJ_RD'] = (
      np.nanmean(final_aj_rd_vals)
      if any(~np.isnan(v) for v in final_aj_rd_vals)
      else float('nan')
  )
  return metrics

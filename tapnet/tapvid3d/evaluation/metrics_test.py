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
"""Regression tests for TAPVid-3D depth scaling (issue #173)."""

import unittest

import numpy as np

from tapnet.tapvid3d.evaluation import metrics


class ComputeTapvid3dMetricsTest(unittest.TestCase):

  @staticmethod
  def _sample(order='t n', batched=False):
    rng = np.random.default_rng(0)
    tracks = rng.uniform(1.0, 5.0, size=(10, 4, 3))
    occluded = np.zeros((10, 4), dtype=bool)
    query_points = np.zeros((4, 3), dtype=float)

    if order in ('n t', 'b n t'):
      tracks = np.swapaxes(tracks, 0, 1)
      occluded = np.swapaxes(occluded, 0, 1)

    if batched:
      tracks = tracks[np.newaxis]
      occluded = occluded[np.newaxis]
      query_points = query_points[np.newaxis]

    return {
        'gt_occluded': occluded,
        'gt_tracks': tracks,
        'pred_occluded': occluded,
        'pred_tracks': tracks,
        'intrinsics_params': np.array([100.0, 100.0, 50.0, 50.0]),
        'query_points': query_points,
        'order': order,
    }

  def test_unbatched_video_metrics_for_perfect_predictions(self):
    for scaling in ('median', 'per_trajectory', 'local_neighborhood'):
      for order in ('n t', 't n'):
        for use_fixed_threshold in (False, True):
          with self.subTest(
              scaling=scaling,
              order=order,
              use_fixed_threshold=use_fixed_threshold,
          ):
            result = metrics.compute_tapvid3d_metrics(
                **self._sample(order=order),
                scaling=scaling,
                use_fixed_metric_threshold=use_fixed_threshold,
            )
            for name in (
                'average_jaccard',
                'average_pts_within_thresh',
                'occlusion_accuracy',
            ):
              self.assertEqual(result[name].shape, (1,))
              np.testing.assert_allclose(result[name], [1.0])

  def test_unbatched_trackwise_metrics_for_perfect_predictions(self):
    for scaling in ('median', 'per_trajectory', 'local_neighborhood'):
      for order in ('n t', 't n'):
        with self.subTest(scaling=scaling, order=order):
          result = metrics.compute_tapvid3d_metrics(
              **self._sample(order=order),
              scaling=scaling,
              get_trackwise_metrics=True,
          )
          expected_shape = (
              (4,) if scaling == 'local_neighborhood' else (1, 4)
          )
          self.assertEqual(result['average_jaccard'].shape, expected_shape)
          np.testing.assert_allclose(result['average_jaccard'], 1.0)

  def test_existing_batched_modes_remain_unchanged(self):
    for scaling in ('median', 'per_trajectory'):
      for order in ('b n t', 'b t n'):
        with self.subTest(scaling=scaling, order=order):
          result = metrics.compute_tapvid3d_metrics(
              **self._sample(order=order, batched=True),
              scaling=scaling,
          )
          self.assertEqual(result['average_jaccard'].shape, (1,))
          np.testing.assert_allclose(result['average_jaccard'], [1.0])

  def test_local_neighborhood_rejects_batched_inputs(self):
    with self.assertRaisesRegex(
        AssertionError, "Local neighborhood doesn't support batched inputs."
    ):
      metrics.compute_tapvid3d_metrics(
          **self._sample(order='b t n', batched=True),
          scaling='local_neighborhood',
      )

  def test_unbatched_scaling_recovers_constant_depth_scale(self):
    for scaling in ('per_trajectory', 'local_neighborhood'):
      for order in ('n t', 't n'):
        with self.subTest(scaling=scaling, order=order):
          data = self._sample(order=order)
          data['query_points'][:, 0] = np.arange(4)
          data['pred_tracks'] = data['pred_tracks'] * 2.0
          result = metrics.compute_tapvid3d_metrics(**data, scaling=scaling)
          self.assertEqual(result['average_jaccard'].shape, (1,))
          np.testing.assert_allclose(result['average_jaccard'], [1.0])


if __name__ == '__main__':
  unittest.main()

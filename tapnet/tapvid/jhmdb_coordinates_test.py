# SPDX-License-Identifier: Apache-2.0
"""Regression tests for query coordinates in the JHMDB loader."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import scipy.io as sio

from tapnet.tapvid import evaluation_datasets


class JhmdbCoordinatesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.root = Path(tmp.name)

    splits = self.root / 'splits'
    splits.mkdir()
    (splits / 'walk_test_split1.txt').write_text('clip.avi 2\n')

    clip = Path('walk') / 'clip'
    joints = self.root / 'joint_positions' / clip
    joints.mkdir(parents=True)
    # JHMDB stores (x/y, joint, frame). Give the two frames different poses.
    pose = np.array([[[3.0, 7.0]], [[2.0, 5.0]]])
    sio.savemat(joints / 'joint_positions.mat', {'pos_img': pose})

    images = self.root / 'Rename_Images' / clip
    images.mkdir(parents=True)
    for name in ('00001.png', '00002.png'):
      Image.new('RGB', (8, 6)).save(images / name)

  def test_resized_query_uses_first_frame_target_coordinates(self):
    dataset = evaluation_datasets.create_jhmdb_dataset(str(self.root), (4, 4))
    sample = next(dataset)['jhmdb']
    self.assertEqual(sample['video'].shape, (1, 2, 4, 4, 3))
    np.testing.assert_allclose(sample['gt_pose_orig'][0, 0, 0], [3, 2])
    np.testing.assert_allclose(sample['target_points'][0, 0, 0], [1.5, 4 / 3])
    np.testing.assert_allclose(sample['query_points'][0, 0], [0, 4 / 3, 1.5])

  def test_default_resolution(self):
    dataset = evaluation_datasets.create_jhmdb_dataset(str(self.root))
    sample = next(dataset)['jhmdb']
    self.assertEqual(sample['video'].shape, (1, 2, 256, 256, 3))
    np.testing.assert_allclose(sample['query_points'][0, 0], [0, 256 / 3, 96])

  def test_original_resolution(self):
    dataset = evaluation_datasets.create_jhmdb_dataset(
        str(self.root), resolution=None
    )
    sample = next(dataset)['jhmdb']
    self.assertEqual(sample['video'].shape, (1, 2, 6, 8, 3))
    np.testing.assert_allclose(sample['target_points'][0, 0, 0], [3, 2])
    np.testing.assert_allclose(sample['query_points'][0, 0], [0, 2, 3])


if __name__ == '__main__':
  unittest.main()

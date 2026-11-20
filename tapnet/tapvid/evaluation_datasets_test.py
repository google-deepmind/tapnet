# SPDX-License-Identifier: Apache-2.0
"""Tests for the TAP-Vid evaluation dataset loaders."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import scipy.io as sio

from tapnet.tapvid import evaluation_datasets


class JhmdbDatasetTest(unittest.TestCase):

  def test_loads_sample_from_supplied_root(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      clip = Path('walk') / 'clip'
      splits = root / 'splits'
      splits.mkdir()
      (splits / 'walk_test_split1.txt').write_text('clip.avi 2\n')

      joints = root / 'joint_positions' / clip
      joints.mkdir(parents=True)
      pose = np.array([3.0, 2.0]).reshape(2, 1, 1)  # x/y, joint, frame
      sio.savemat(joints / 'joint_positions.mat', {'pos_img': pose})

      images = root / 'Rename_Images' / clip
      images.mkdir(parents=True)
      Image.new('RGB', (8, 6)).save(images / '00001.png')

      dataset = evaluation_datasets.create_jhmdb_dataset(tmp, resolution=None)
      sample = next(dataset)['jhmdb']
      self.assertEqual(sample['fname'], str(clip))
      self.assertEqual(sample['video'].shape, (1, 1, 6, 8, 3))
      np.testing.assert_array_equal(sample['target_points'], [[[[3.0, 2.0]]]])


if __name__ == '__main__':
  unittest.main()

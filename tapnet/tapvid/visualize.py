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

"""Visualize frames of a random video of the given dataset."""

import io
import pickle
import random
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import mediapy as media
import numpy as np
from PIL import Image

from tapnet.utils import viz_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', None, 'Path to the pickle file.', required=True
)
flags.DEFINE_string(
    'output_path', None, 'Path to the output mp4 video.', required=True
)


def main(argv: Sequence[str]) -> None:
  del argv

  logging.info('Loading data from %s. This takes time.', FLAGS.input_path)
  with open(FLAGS.input_path, 'rb') as f:
    data = pickle.load(f)
    if isinstance(data, dict):
      data = list(data.values())

  idx = random.randint(0, len(data) - 1)
  video = data[idx]

  frames = video['video']

  if isinstance(frames[0], bytes):
    # Tapnet is stored and JPEG bytes rather than `np.ndarray`s.
    def decode(frame):
      byteio = io.BytesIO(frame)
      img = Image.open(byteio)
      return np.array(img)

    frames = np.array([decode(frame) for frame in frames])

  if frames.shape[1] > 360:
    frames = media.resize_video(frames, (360, 640))

  scale_factor = np.array(frames.shape[2:0:-1])[np.newaxis, np.newaxis, :]
  painted_frames = viz_utils.paint_point_track(
      frames,
      video['points'] * scale_factor,
      ~video['occluded'],
  )

  media.write_video(FLAGS.output_path, painted_frames, fps=25)
  logging.info('Examplar point visualization saved to %s', FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)

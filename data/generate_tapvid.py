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

"""Python script to generate a pickle with annotations from raw videos."""

import csv
import dataclasses
import io
import math
import os
import pickle
from typing import Dict, Iterator, List, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import ffmpeg
import numpy as np
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string('input_csv_path', None, 'Path to the input csv.')
flags.DEFINE_string(
    'output_base_path',
    None,
    'Path to the output folder where pickle files will be stored.',
)
flags.DEFINE_string(
    'video_root_path',
    None,
    (
        'Path to the root directory of the extracted Kinetics-700-2020 videos'
        ' from the validation set.'
    ),
)
flags.DEFINE_integer('num_shards', 10, 'Number of pickle shards to output.')

_JPEG_HEADER = b'\xff\xd8'


@dataclasses.dataclass(frozen=True)
class Point:
  x: float
  y: float
  occluded: bool


@dataclasses.dataclass(frozen=True)
class Track:
  points: Tuple[Point, ...]


@dataclasses.dataclass(frozen=True)
class Video:
  youtube_id: str
  start_time_sec: int
  end_time_sec: int
  video_path: str
  tracks: Tuple[Track, ...]


def csv_to_dataset(
    csv_path: str, videos_path: Dict[str, str]
) -> Tuple[Video, ...]:
  """Reads the input CSV and creates a list of `Video`s out of that."""

  def points(row: Sequence[str]) -> Iterator[Point]:
    for i in range(250):
      x, y, occ = row[3 + 3 * i : 3 + 3 * i + 3]
      x = float(x)
      y = float(y)
      assert occ in ('0', '1')
      occ = occ == '1'
      yield Point(x, y, occ)

  logging.info('Reading CSV "%s".', csv_path)

  with open(csv_path) as f:
    reader = csv.reader(f, delimiter=',')

    tracks_per_video: Dict[Tuple[str, int, int], List[Track]] = {}
    for row in reader:
      assert len(row) == 3 + 3 * 250

      youtube_id, start_time_sec, end_time_sec = row[:3]
      start_time_sec = int(start_time_sec)
      end_time_sec = int(end_time_sec)
      key = (youtube_id, start_time_sec, end_time_sec)

      track = Track(tuple(points(row)))

      if key not in tracks_per_video:
        tracks_per_video[key] = []
      tracks_per_video[key].append(track)

    def videos() -> Iterator[Video]:
      for key, tracks in tracks_per_video.items():
        youtube_id, start_time_sec, end_time_sec = key

        name = f'{youtube_id}_{start_time_sec:06}_{end_time_sec:06}'
        if name not in videos_path:
          logging.warning('Video "%s" not downloaded. Skipping it.', name)
          continue
        video_path = videos_path[name]

        yield Video(
            youtube_id, start_time_sec, end_time_sec, video_path, tuple(tracks)
        )

    return tuple(videos())


def get_paths_to_videos(video_root_path: str) -> Dict[str, str]:
  """Returns the relative path to each downloaded video."""
  logging.info('Reading all videos in subfolders of "%s".', video_root_path)
  video_to_path: Dict[str, str] = {}
  for folder_or_video in os.listdir(video_root_path):
    path = os.path.join(video_root_path, folder_or_video)
    if os.path.isdir(path):
      subfolder_paths = get_paths_to_videos(path)
      for k, v in subfolder_paths.items():
        assert k not in video_to_path
        video_to_path[k] = v
    elif folder_or_video.endswith('.mp4'):
      name = folder_or_video[:-4]  # Remove '.mp4'.
      assert name not in video_to_path
      video_to_path[name] = path

  return video_to_path


def extract_frames(video_path: str, fps: float) -> Tuple[bytes, ...]:
  """Extracts list of jpeg bytes from the given video using ffmpeg."""
  cmd = (
      ffmpeg.input(video_path)
      .filter('fps', fps=fps)
      .output('pipe:', format='image2pipe')
  )
  jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=True)
  jpeg_bytes = jpeg_bytes.split(_JPEG_HEADER)[1:]
  jpeg_bytes = map(lambda x: _JPEG_HEADER + x, jpeg_bytes)
  return tuple(jpeg_bytes)


def generate_example(video: Video) -> Dict[str, np.ndarray]:
  """Generates a dictionary with the info from a `Video`."""
  example: Dict[str, np.ndarray] = {}

  imgs_encoded = extract_frames(video.video_path, 25.0)
  if len(imgs_encoded) > 250:
    imgs_encoded = imgs_encoded[:250]

  if len(imgs_encoded) < 250:
    # Clip is shorter than 10s.
    num_frames = len(imgs_encoded)
    new_tracks = tuple(
        Track(tuple(t.points[:num_frames])) for t in video.tracks
    )
    video = Video(
        video.youtube_id,
        video.start_time_sec,
        video.end_time_sec,
        video.video_path,
        new_tracks,
    )

  example['video'] = np.array(imgs_encoded)
  byteio = io.BytesIO(imgs_encoded[0])
  img = Image.open(byteio)
  height, width, _ = np.array(img).shape

  points = []
  occluded = []
  for track in video.tracks:
    points.append(
        [
            [(p.x * width - 0.5) / width, (p.y * height - 0.5) / height]
            for p in track.points
        ]
    )
    occluded.append([p.occluded for p in track.points])

  example['points'] = np.array(points, dtype=np.float64)
  example['occluded'] = np.array(occluded, dtype=bool)

  return example


def main(argv: Sequence[str]) -> None:
  del argv

  output_folder = FLAGS.output_base_path
  if output_folder and not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Reads data.
  videos_path = get_paths_to_videos(FLAGS.video_root_path)
  videos = csv_to_dataset(FLAGS.input_csv_path, videos_path)

  # Process the dataset and store pickles.
  num_examples_per_shard = int(math.ceil(len(videos) / FLAGS.num_shards))
  shard = 0
  data = []
  for i, video in enumerate(videos):
    print(
        'Processing example %d of %d   (%d%%) \r'
        % (i, len(videos), i * 100 / len(videos)),
        end='',
    )
    data.append(generate_example(video))
    if i == len(videos) - 1 or len(data) == num_examples_per_shard:
      shard_path = os.path.join(
          output_folder, f'{shard:04}_of_{FLAGS.num_shards:04}.pkl'
      )
      logging.info('Writing file "%s".', shard_path)
      with open(shard_path, 'wb') as f:
        pickle.dump(data, f)
      data.clear()
      shard += 1


if __name__ == '__main__':
  app.run(main)

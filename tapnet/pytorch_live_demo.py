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

"""Live Demo for PyTorch Online TAPIR."""

import time

import cv2
import numpy as np
from tapnet.torch import tapir_model
import torch
import torch.nn.functional as F
import tree

NUM_POINTS = 8


def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames


def online_model_init(frames, points):
  """Initialize query features for the query points."""
  frames = preprocess_frames(frames)
  feature_grids = model.get_feature_grids(frames, is_training=False)
  features = model.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features


def postprocess_occlusions(occlusions, expected_dist):
  visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
  return visibles


def online_model_predict(frames, features, causal_context):
  """Compute point tracks and occlusions given frames and query points."""
  frames = preprocess_frames(frames)
  feature_grids = model.get_feature_grids(frames, is_training=False)
  trajectories = model.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )
  causal_context = trajectories["causal_context"]
  del trajectories["causal_context"]
  # Take only the predictions for the final resolution.
  # For running on higher resolution, it's typically better to average across
  # resolutions.
  tracks = trajectories["tracks"][-1]
  occlusions = trajectories["occlusion"][-1]
  uncertainty = trajectories["expected_dist"][-1]
  visibles = postprocess_occlusions(occlusions, uncertainty)
  return tracks, visibles, causal_context


def get_frame(video_capture):
  r_val, image = video_capture.read()
  trunc = np.abs(image.shape[1] - image.shape[0]) // 2
  if image.shape[1] > image.shape[0]:
    image = image[:, trunc:-trunc]
  elif image.shape[1] < image.shape[0]:
    image = image[trunc:-trunc]
  return r_val, image


print("Welcome to the TAPIR PyTorch live demo.")
print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
print("may degrade and you may need a more powerful GPU.")

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

# --------------------
# Load checkpoint and initialize
print("Creating model...")
model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
print("Loading checkpoint...")
model.load_state_dict(
    torch.load("tapnet/checkpoints/causal_bootstapir_checkpoint.pt")
)
model = model.to(device)
model = model.eval()
torch.set_grad_enabled(False)

# --------------------
# Start point tracking
print("Initializing camera...")
vc = cv2.VideoCapture(0)

vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if vc.isOpened():  # try to get the first frame
  rval, frame = get_frame(vc)
else:
  raise ValueError("Unable to open camera.")

pos = tuple()
query_frame = True
have_point = [False] * NUM_POINTS

query_points = torch.zeros([NUM_POINTS, 3], dtype=torch.float32)
query_points = query_points.to(device)
frame = torch.tensor(frame).to(device)

query_features = online_model_init(
    frames=frame[None, None], points=query_points[None, :]
)

causal_state = model.construct_initial_causal_state(
    NUM_POINTS, len(query_features.resolutions) - 1
)
causal_state = tree.map_structure(lambda x: x.to(device), causal_state)

prediction, visible, causal_state = online_model_predict(
    frames=frame[None, None],
    features=query_features,
    causal_context=causal_state,
)

next_query_idx = 0
last_click_time = 0


def mouse_click(event, x, y, flags, param):
  del flags, param
  global pos, query_frame, last_click_time

  # event fires multiple times per click sometimes??
  if (time.time() - last_click_time) < 0.5:
    return

  if event == cv2.EVENT_LBUTTONDOWN:
    pos = (y, frame.shape[1] - x)
    query_frame = True
    last_click_time = time.time()


cv2.namedWindow("Point Tracking")
cv2.setMouseCallback("Point Tracking", mouse_click)

t = time.time()
step_counter = 0

print("Press ESC to exit.")

while rval:
  rval, frame = get_frame(vc)
  numpy_frame = frame
  if query_frame:
    query_points = np.array((0,) + pos, dtype=np.float32)
    frame = torch.tensor(frame).to(device)
    query_points = torch.tensor(query_points).to(device)

    init_query_features = online_model_init(
        frames=frame[None, None], points=query_points[None, None]
    )
    query_frame = False
    query_features, causal_state = model.update_query_features(
        query_features=query_features,
        new_query_features=init_query_features,
        idx_to_update=np.array([next_query_idx]),
        causal_state=causal_state,
    )
    have_point[next_query_idx] = True
    next_query_idx = (next_query_idx + 1) % NUM_POINTS
  if pos:
    frame = torch.tensor(frame).to(device)
    track, visible, causal_state = online_model_predict(
        frames=frame[None, None],
        features=query_features,
        causal_context=causal_state,
    )
    track = np.round(track.cpu().numpy())
    visible = visible.cpu().numpy()

    for i, _ in enumerate(have_point):
      if visible[0, i, 0] and have_point[i]:
        cv2.circle(
            numpy_frame,
            (int(track[0, i, 0, 0]), int(track[0, i, 0, 1])),
            5,
            (255, 0, 0),
            -1,
        )
        if track[0, i, 0, 0] < 16 and track[0, i, 0, 1] < 16:
          print((i, next_query_idx))
  cv2.imshow("Point Tracking", numpy_frame[:, ::-1])
  if pos:
    step_counter += 1
    if time.time() - t > 5:
      print(f"{step_counter/(time.time()-t)} frames per second")
      t = time.time()
      step_counter = 0
  else:
    t = time.time()
  key = cv2.waitKey(1)

  if key == 27:  # exit on ESC
    break

cv2.destroyWindow("Point Tracking")
vc.release()

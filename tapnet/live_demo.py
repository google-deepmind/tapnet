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

"""Live Demo for Online TAPIR."""

import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils


NUM_POINTS = 8


def load_checkpoint(checkpoint_path):
  ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
  return ckpt_state["params"], ckpt_state["state"]

print("Loading checkpoint...")
# --------------------
# Load checkpoint and initialize
params, state = load_checkpoint(
    "tapnet/checkpoints/causal_tapir_checkpoint.npy"
)

tapir = tapir_model.ParameterizedTAPIR(
    params=params,
    state=state,
    tapir_kwargs=dict(
        use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
    ),
)


def online_model_init(frames, points):
  feature_grids = tapir.get_feature_grids(frames, is_training=False)
  features = tapir.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features


def online_model_predict(frames, features, causal_context):
  """Compute point tracks and occlusions given frames and query points."""
  feature_grids = tapir.get_feature_grids(frames, is_training=False)
  trajectories = tapir.estimate_trajectories(
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
  return {k: v[-1] for k, v in trajectories.items()}, causal_context


def get_frame(video_capture):
  r_val, image = video_capture.read()
  trunc = np.abs(image.shape[1] - image.shape[0]) // 2
  if image.shape[1] > image.shape[0]:
    image = image[:, trunc:-trunc]
  elif image.shape[1] < image.shape[0]:
    image = image[trunc:-trunc]
  return r_val, image


print("Welcome to the TAPIR live demo.")
print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
print("may degrade and you may need a more powerful GPU.")

print("Creating model...")
online_init_apply = jax.jit(online_model_init)

online_predict_apply = jax.jit(online_model_predict)

print("Initializing camera...")
# --------------------
# Start point tracking
vc = cv2.VideoCapture(0)

vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if vc.isOpened():  # try to get the first frame
  rval, frame = get_frame(vc)
else:
  raise ValueError("Unable to open camera.")

pos = tuple()
query_frame = True
have_point = [False] * NUM_POINTS
query_features = None
causal_state = None
next_query_idx = 0

print("Compiling jax functions (this may take a while...)")
# --------------------
# Call one time to compile
query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
_ = online_init_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    points=query_points[None, 0:1],
)
jax.block_until_ready(query_features)
query_features = online_init_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    points=query_points[None, :],
)

causal_state = tapir.construct_initial_causal_state(
    NUM_POINTS, len(query_features.resolutions) - 1
)

prediction, causal_state = online_predict_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    features=query_features,
    causal_context=causal_state,
)

jax.block_until_ready(prediction["tracks"])

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
  if query_frame:
    query_points = jnp.array((0,) + pos, dtype=jnp.float32)

    init_query_features = online_init_apply(
        frames=model_utils.preprocess_frames(frame[None, None]),
        points=query_points[None, None],
    )
    query_frame = False
    query_features, causal_state = tapir.update_query_features(
        query_features=query_features,
        new_query_features=init_query_features,
        idx_to_update=np.array([next_query_idx]),
        causal_state=causal_state,
    )
    have_point[next_query_idx] = True
    next_query_idx = (next_query_idx + 1) % NUM_POINTS
  if pos:
    prediction, causal_state = online_predict_apply(
        frames=model_utils.preprocess_frames(frame[None, None]),
        features=query_features,
        causal_context=causal_state,
    )
    track = prediction["tracks"][0, :, 0]
    occlusion = prediction["occlusion"][0, :, 0]
    expected_dist = prediction["expected_dist"][0, :, 0]
    visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
    track = np.round(track)

    for i, _ in enumerate(have_point):
      if visibles[i] and have_point[i]:
        cv2.circle(
            frame, (int(track[i, 0]), int(track[i, 1])), 5, (255, 0, 0), -1
        )
        if track[i, 0] < 16 and track[i, 1] < 16:
          print((i, next_query_idx))
  cv2.imshow("Point Tracking", frame[:, ::-1])
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

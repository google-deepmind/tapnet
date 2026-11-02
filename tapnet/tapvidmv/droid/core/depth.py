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

"""Stereo disparity estimation and SAM-based wrist gripper depth refinement."""

import core.geometry
import cv2
import numpy as np
import torch
from tqdm import tqdm


@torch.inference_mode()
def get_s2m2_disparity(img_left, img_right, s2m2_model, device, conf_thresh):
  """Estimates confidence-filtered stereo disparity for a stereo RGB pair."""
  # pylint: disable=g-import-not-at-top
  from s2m2.core.utils.image_utils import image_crop
  from s2m2.core.utils.image_utils import image_pad
  # pylint: enable=g-import-not-at-top

  height, width = img_left.shape[:2]
  left_torch = image_pad(
      torch.from_numpy(img_left).permute(2, 0, 1).unsqueeze(0).to(device), 32
  )
  right_torch = image_pad(
      torch.from_numpy(img_right).permute(2, 0, 1).unsqueeze(0).to(device), 32
  )

  pred_disparity, _, pred_confidence = s2m2_model(left_torch, right_torch)

  disparity = (
      image_crop(pred_disparity, (height, width)).cpu().numpy().squeeze()
  )
  confidence = (
      image_crop(pred_confidence, (height, width)).cpu().numpy().squeeze()
  )

  valid_mask = (disparity > 0) & (confidence >= conf_thresh)
  disparity[~valid_mask] = 0.0

  return disparity


def compute_stereo_depth(episode, s2m2_model, device, conf_thresh):
  """Computes metric stereo depth maps across all cameras in an episode."""
  for cam_id, cam_data in episode["camera"].items():
    left_seq, right_seq = cam_data["video_rgb"], cam_data["video_right"]
    raw_disparity = np.stack([
        get_s2m2_disparity(img_left, img_right, s2m2_model, device, conf_thresh)
        for img_left, img_right in tqdm(
            zip(left_seq, right_seq),
            total=len(left_seq),
            desc=f"Depth [{cam_id}]",
        )
    ])
    cam_data["raw_depth"] = core.geometry.decode_disparity(
        raw_disparity, cam_data["K"][0, 0], cam_data["baseline"]
    )
  return episode


def extract_single_frame_mask(img_rgb, predictor, mask_area_min, mask_area_max):
  """Extracts a wrist-gripper foreground mask from one RGB frame using SAM."""
  height, width = img_rgb.shape[:2]

  points = np.array([
      [width // 2 - 120, height - 110],
      [width // 2 + 500, height - 110],
      [width // 2 - 250, height - 25],
      [width // 2 + 450, height - 25],
      [width // 2 + 100, height - 15],
      [width // 2 + 100, height - 300],
  ])
  labels = np.array([1, 1, 1, 1, 1, 0])
  bbox = np.array([0, height // 2, width, height])

  predictor.set_image(img_rgb)
  masks, scores, _ = predictor.predict(
      point_coords=points, point_labels=labels, box=bbox, multimask_output=True
  )

  valid_masks, valid_scores = [], []
  for m, s in zip(masks, scores):
    area_ratio = np.sum(m) / (width * height)
    if mask_area_min < area_ratio < mask_area_max:
      valid_masks.append(m)
      valid_scores.append(s * area_ratio)

  if valid_masks:
    best_mask = valid_masks[np.argmax(valid_scores)]
  else:
    best_mask = masks[np.argmax(scores)]

  return best_mask


def compute_consensus_mask(masks_list, consensus_thresh):
  vote_map = np.mean(masks_list, axis=0, dtype=np.float32)
  consensus_mask = vote_map >= consensus_thresh

  n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
      consensus_mask.astype(np.uint8)
  )
  if n_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    consensus_mask = labels_map == largest_label

  return consensus_mask


def refine_gripper_depth(episode, sam_predictor, config):
  """Distills and injects stable open-gripper surface depth on the wrist view."""
  cam_data = episode["camera"][episode["meta"]["wrist_cam_id"]]
  raw_depth = cam_data["raw_depth"]
  cam_data["original_raw_depth"] = raw_depth.copy()
  cam_data["sam_real_masks"] = np.zeros(raw_depth.shape, dtype=bool)
  gripper_depth = np.zeros_like(raw_depth[0])

  open_frames = np.flatnonzero(
      episode["robot"]["gripper_positions"].reshape(-1)
      < config.gripper_open_thresh
  )
  if open_frames.size > 0:
    masks = [
        extract_single_frame_mask(
            cam_data["video_rgb"][t],
            sam_predictor,
            config.mask_area_min,
            config.mask_area_max,
        )
        for t in tqdm(open_frames, desc="SAM mask")
    ]
    mask = compute_consensus_mask(masks, config.consensus_thresh)
    cam_data["sam_real_masks"][open_frames] = mask

    open_depth = raw_depth[open_frames]
    samples = np.where(
        mask & (open_depth > 0) & (open_depth < config.max_depth_thresh),
        open_depth,
        np.nan,
    )
    observed = ~np.isnan(samples).all(axis=0)
    if observed.any():
      gripper_depth[observed] = np.nanmedian(samples[:, observed], axis=0)
    raw_depth[open_frames] = np.where(
        gripper_depth > 0, gripper_depth, open_depth
    )

  cam_data["empirical_gripper_depth"] = gripper_depth
  return episode

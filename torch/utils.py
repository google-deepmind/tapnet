# Copyright 2024 DeepMind Technologies Limited
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

"""Pytorch model utilities."""

from typing import Any, Union, Optional, List
#from einshape.src import abstract_ops
#from einshape.src import backend
import numpy as np
import torch
import torch.nn.functional as F
import math

def bilinear(x: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
  """Resizes a 5D tensor using bilinear interpolation.

  Args:
        x: A 5D tensor of shape (B, T, W, H, C) where B is batch size, T is
          time, W is width, H is height, and C is the number of channels.
    resolution: The target resolution as a tuple (new_width, new_height).

  Returns:
    The resized tensor.
  """
  b, t, h, w, c = x.size()
  x = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
  x = F.interpolate(x, size=resolution, mode='bilinear', align_corners=False)
  b, _, h, w = x.size()
  x = x.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)
  return x


def map_coordinates_3d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 3D coordinates to corresponding features using bilinear interpolation.

  Args:
    feats: A 5D tensor of features with shape (B, W, H, D, C), where B is batch
      size, W is width, H is height, D is depth, and C is the number of
      channels.
    coordinates: A 3D tensor of coordinates with shape (B, N, 3), where N is the
      number of coordinates and the last dimension represents (W, H, D)
      coordinates.

  Returns:
    The mapped features tensor.
  """
  x = feats.permute(0, 4, 1, 2, 3)
  y = coordinates[:, :, None, None, :].float()
  y[..., 0] += 0.5
  y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
  y = torch.flip(y, dims=(-1,))
  out = (
      F.grid_sample(
          x, y, mode='bilinear', align_corners=False, padding_mode='border'
      )
      .squeeze(dim=(3, 4))
      .permute(0, 2, 1)
  )
  return out


def map_coordinates_2d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 2D coordinates to feature maps using bilinear interpolation.

  The function performs bilinear interpolation on the feature maps (`feats`)
  at the specified `coordinates`. The coordinates are normalized between
  -1 and 1 The result is a tensor of sampled features corresponding
  to these coordinates.

  Args:
    feats (Tensor): A 5D tensor of shape (N, T, H, W, C) representing feature
      maps, where N is the batch size, T is the number of frames, H and W are
      height and width, and C is the number of channels.
    coordinates (Tensor): A 5D tensor of shape (N, P, T, S, XY) representing
      coordinates, where N is the batch size, P is the number of points, T is
      the number of frames, S is the number of samples, and XY represents the 2D
      coordinates.

  Returns:
    Tensor: A 5D tensor of the sampled features corresponding to the
      given coordinates, of shape (N, P, T, S, C).
  """
  n, t, h, w, c = feats.shape
  x = feats.permute(0, 1, 4, 2, 3).view(n * t, c, h, w)

  n, p, t, s, xy = coordinates.shape
  y = coordinates.permute(0, 2, 1, 3, 4).view(n * t, p, s, xy)
  y = 2 * (y / h) - 1
  y = torch.flip(y, dims=(-1,)).float()

  out = F.grid_sample(
      x, y, mode='bilinear', align_corners=False, padding_mode='zeros'
  )
  _, c, _, _ = out.shape
  out = out.permute(0, 2, 3, 1).view(n, t, p, s, c).permute(0, 2, 1, 3, 4)

  return out


def soft_argmax_heatmap_batched(softmax_val, threshold:float=5):
  """Test if two image resolutions are the same."""
  b, h, w, d1, d2 = softmax_val.shape
  y, x = torch.meshgrid(
      torch.arange(d1, device=softmax_val.device),
      torch.arange(d2, device=softmax_val.device),
      indexing='ij',
  )
  coords = torch.stack([x + 0.5, y + 0.5], dim=-1).to(softmax_val.device)
  softmax_val_flat = softmax_val.reshape(b, h, w, -1)
  argmax_pos = torch.argmax(softmax_val_flat, dim=-1)

  pos = coords.reshape(-1, 2)[argmax_pos]
  
  tmp1 = torch.square(coords[None, None, None, :, :, :] - pos[:, :, :, None, None, :])
  tmp2 = torch.unsqueeze(torch.sum(tmp1, dim=-1), -1)
  valid = (tmp2 < threshold**2)

  weighted_sum = torch.sum(
      coords[None, None, None, :, :, :]
      * valid
      * softmax_val[:, :, :, :, :, None],
      dim=(3, 4),
  )
  sum_of_weights = torch.maximum(
      torch.sum(valid * softmax_val[:, :, :, :, :, None], dim=(3, 4)),
      torch.tensor(1e-12, device=softmax_val.device),
  )
  return weighted_sum / sum_of_weights


def heatmaps_to_points(
    all_pairs_softmax,
    image_shape:List[int],
    threshold:float=5,
    query_points:Optional[torch.Tensor]=None,
):
  """Convert heatmaps to points using soft argmax."""

  out_points = soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
  feature_grid_shape = all_pairs_softmax.shape[1:]
  # Note: out_points is now [x, y]; we need to divide by [width, height].
  # image_shape[3] is width and image_shape[2] is height.
  t1 = out_points.detach()
  out_points = convert_grid_coordinates(
      t1,
      torch.tensor(feature_grid_shape[3:1:-1], device=t1.device),
      torch.tensor(image_shape[3:1:-1], device=t1.device),
  )
  assert feature_grid_shape[1] == image_shape[1]
  if query_points is not None:
    # The [..., 0:1] is because we only care about the frame index.
    t2 = query_points.detach()
    query_frame = convert_grid_coordinates(
        t2,
        torch.tensor(image_shape[1:4], device=t2.device),
        torch.tensor(feature_grid_shape[1:4], device=t2.device),
        coordinate_format='tyx',
    )[..., 0:1]

    query_frame = torch.round(query_frame)
    frame_indices = torch.arange(image_shape[1], device=query_frame.device)[
        None, None, :
    ]
    is_query_point = query_frame == frame_indices

    is_query_point = is_query_point[:, :, :, None]
    out_points = (
        out_points * ~is_query_point
        + torch.flip(query_points[:, :, None], dims=(-1,))[..., 0:2]
        * is_query_point
    )

  return out_points


def is_same_res(r1 : tuple[int,int], r2 : tuple[int,int]):
  """Test if two image resolutions are the same."""
  #return all([x == y for x, y in zip(r1, r2)])
  return (r1[0] == r2[0]) and (r1[1] == r2[1])


def convert_grid_coordinates(
    coords: torch.Tensor,
    input_grid_size, # Sequence not supported by TorchScript, caller must convert torch.size or a tuple to torch.tensor
    output_grid_size,# Sequence not supported by TorchScript, caller must convert torch.size or a tuple to torch.tensor
    coordinate_format: str = 'xy',
) -> torch.Tensor:
  """Convert grid coordinates to correct format."""
  if isinstance(input_grid_size, tuple):
    input_grid_size = torch.tensor(input_grid_size, device=coords.device)
  if isinstance(output_grid_size, tuple):
    output_grid_size = torch.tensor(output_grid_size, device=coords.device)
  if not isinstance(input_grid_size, torch.Tensor):
    raise ValueError(f"input_grid_size must be a torch.tensor, not {type(input_grid_size)}")
  if not isinstance(output_grid_size, torch.Tensor):
    raise ValueError(f"output_grid_size must be a torch.tensor, not {type(output_grid_size)}")

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.'
      )
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.'
      )
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid

def generate_default_resolutions(full_size : tuple[int,int], train_size : tuple[int,int]): #, num_levels : int = None):
  """Generate a list of logarithmically-spaced resolutions.

  Generated resolutions are between train_size and full_size, inclusive, with
  num_levels different resolutions total.  Useful for generating the input to
  refinement_resolutions in PIPs.

  Args:
    full_size: 2-tuple of ints.  The full image size desired.
    train_size: 2-tuple of ints.  The smallest refinement level.  Should
      typically match the training resolution, which is (256, 256) for TAPIR.
    num_levels: number of levels.  Typically each resolution should be less than
      twice the size of prior resolutions.

  Returns:
    A list of resolutions.
  """
  if all([x == y for x, y in zip(train_size, full_size)]):
    return [train_size]

  ratio0 = full_size[0] / train_size[0]
  ratio1 = full_size[1] / train_size[1]
  if ratio1 > ratio0:
    ratio0 = ratio1
  tr = torch.tensor([ratio0])
  tr = torch.ceil(torch.log2(tr))
  num_levels = int(tr.item() + 1)

  if num_levels <= 1:
    return [train_size]

  h, w = full_size[0:2]
  if h % 8 != 0 or w % 8 != 0:
    print(
        'Warning: output size is not a multiple of 8. Final layer '
        + 'will round size down.'
    )
  ll_h = int(train_size[0])
  ll_w = int(train_size[1])

  sizes = [(int(0), int(0))] # in the torch.jit.script() context, annotated assignments without assigned value aren't supported
  sizes.clear()
  for i in range(num_levels):
    size = (
        int(round( float(ll_h * (h / ll_h) ** (i / (num_levels - 1))) // 8 )) * 8,
        int(round( float(ll_w * (w / ll_w) ** (i / (num_levels - 1))) // 8 )) * 8,
    )
    sizes.append(size)
  return sizes


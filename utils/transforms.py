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

"""Utilities for transforming image coordinates."""

from typing import Sequence

import chex
import numpy as np


def convert_grid_coordinates(
    coords: chex.Array,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> chex.Array:
  """Convert image coordinates between image grids of different sizes.

  By default, it assumes that the image corners are aligned.  Therefore,
  it adds .5 (since (0,0) is assumed to be the center of the upper-left grid
  cell), multiplies by the size ratio, and then subtracts .5.

  Args:
    coords: The coordinates to be converted.  It is of shape [..., 2] if
      coordinate_format is 'xy' or [..., 3] if coordinate_format is 'tyx'.
    input_grid_size: The size of the image/grid that the coordinates currently
      are with respect to.  This is a 2-tuple of the format [width, height]
      if coordinate_format is 'xy' or a 3-tuple of the format
      [num_frames, height, width] if coordinate_format is 'tyx'.
    output_grid_size: The size of the target image/grid that you want the
      coordinates to be with respect to.  This is a 2-tuple of the format
      [width, height] if coordinate_format is 'xy' or a 3-tuple of the format
      [num_frames, height, width] if coordinate_format is 'tyx'.
    coordinate_format: Which format the coordinates are in.  This can be one
      of 'xy' (the default) or 'tyx', which are the only formats used in this
      project.

  Returns:
    The transformed coordinates, of the same shape as coordinates.

  Raises:
    ValueError: if coordinates don't match the given format.
  """
  if isinstance(input_grid_size, tuple):
    input_grid_size = np.array(input_grid_size)
  if isinstance(output_grid_size, tuple):
    output_grid_size = np.array(output_grid_size)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.')
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.')
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid

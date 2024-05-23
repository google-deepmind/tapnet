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

"""TAPIR models definition."""

import functools
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from tapnet.torch import nets
from tapnet.torch import utils


class FeatureGrids(NamedTuple):
  """Feature grids for a video, used to compute trajectories.

  These are per-frame outputs of the encoding resnet.

  Attributes:
    lowres: Low-resolution features, one for each resolution; 256 channels.
    hires: High-resolution features, one for each resolution; 64 channels.
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]


class QueryFeatures(NamedTuple):
  """Query features used to compute trajectories.

  These are sampled from the query frames and are a full descriptor of the
  tracked points. They can be acquired from a query image and then reused in a
  separate video.

  Attributes:
    lowres: Low-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 256]
    hires: High-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 64]
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]


class TAPIR(nn.Module):
  """TAPIR model."""

  def __init__(
      self,
      bilinear_interp_with_depthwise_conv: bool = False,
      num_pips_iter: int = 4,
      pyramid_level: int = 1,
      mixer_hidden_dim: int = 512,
      num_mixer_blocks: int = 12,
      mixer_kernel_shape: int = 3,
      patch_size: int = 7,
      softmax_temperature: float = 20.0,
      parallelize_query_extraction: bool = False,
      initial_resolution: Tuple[int, int] = (256, 256),
      blocks_per_group: Sequence[int] = (2, 2, 2, 2),
      feature_extractor_chunk_size: int = 10,
      extra_convs: bool = True,
      use_casual_conv: bool = False,
  ):
    super().__init__()

    self.highres_dim = 128
    self.lowres_dim = 256
    self.bilinear_interp_with_depthwise_conv = (
        bilinear_interp_with_depthwise_conv
    )
    self.parallelize_query_extraction = parallelize_query_extraction

    self.num_pips_iter = num_pips_iter
    self.pyramid_level = pyramid_level
    self.patch_size = patch_size
    self.softmax_temperature = softmax_temperature
    self.initial_resolution = tuple(initial_resolution)
    self.feature_extractor_chunk_size = feature_extractor_chunk_size
    self.num_mixer_blocks = num_mixer_blocks
    self.use_casual_conv = use_casual_conv

    highres_dim = 128
    lowres_dim = 256
    strides = (1, 2, 2, 1)
    blocks_per_group = (2, 2, 2, 2)
    channels_per_group = (64, highres_dim, 256, lowres_dim)
    use_projection = (True, True, True, True)

    self.resnet_torch = nets.ResNet(
        blocks_per_group=blocks_per_group,
        channels_per_group=channels_per_group,
        use_projection=use_projection,
        strides=strides,
    )
    self.torch_cost_volume_track_mods = nn.ModuleDict({
        'hid1': torch.nn.Conv2d(1, 16, 3, 1, 1),
        'hid2': torch.nn.Conv2d(16, 1, 3, 1, 1),
        'hid3': torch.nn.Conv2d(16, 32, 3, 2, 0),
        'hid4': torch.nn.Linear(32, 16),
        'occ_out': torch.nn.Linear(16, 2),
    })
    dim = 4 + self.highres_dim + self.lowres_dim
    input_dim = dim + (self.pyramid_level + 2) * 49
    self.torch_pips_mixer = nets.PIPSMLPMixer(
        input_dim, dim, use_causal_conv=self.use_casual_conv
    )

    if extra_convs:
      self.extra_convs = nets.ExtraConvs()
    else:
      self.extra_convs = None

  def forward(
      self,
      video: torch.Tensor,
      query_points: torch.Tensor,
      is_training: bool = False,
      query_chunk_size: Optional[int] = 64,
      get_query_feats: bool = False,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> Mapping[str, torch.Tensor]:
    """Runs a forward pass of the model.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      query_points: The query points for which we compute tracks.
      is_training: Whether we are training.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      get_query_feats: Return query features for other losses like contrastive.
        Not supported in the current version.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A dict of outputs, including:
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          in raster coordinates
        expected_dist: uncertainty estimate logits, of shape
          [batch, num_queries, num_frames], where higher indicates more likely
          to be far from the correct answer.
    """
    if get_query_feats:
      raise ValueError('Get query feats not supported in TAPIR.')

    feature_grids = self.get_feature_grids(
        video,
        is_training,
        refinement_resolutions,
    )

    query_features = self.get_query_features(
        video,
        is_training,
        query_points,
        feature_grids,
        refinement_resolutions,
    )

    trajectories = self.estimate_trajectories(
        video.shape[-3:-1],
        is_training,
        feature_grids,
        query_features,
        query_points,
        query_chunk_size,
    )

    p = self.num_pips_iter
    out = dict(
        occlusion=torch.mean(
            torch.stack(trajectories['occlusion'][p::p]), dim=0
        ),
        tracks=torch.mean(torch.stack(trajectories['tracks'][p::p]), dim=0),
        expected_dist=torch.mean(
            torch.stack(trajectories['expected_dist'][p::p]), dim=0
        ),
        unrefined_occlusion=trajectories['occlusion'][:-1],
        unrefined_tracks=trajectories['tracks'][:-1],
        unrefined_expected_dist=trajectories['expected_dist'][:-1],
    )

    return out

  def get_query_features(
      self,
      video: torch.Tensor,
      is_training: bool,
      query_points: torch.Tensor,
      feature_grids: Optional[FeatureGrids] = None,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> QueryFeatures:
    """Computes query features, which can be used for estimate_trajectories.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      query_points: The query points for which we compute tracks.
      feature_grids: If passed, we'll use these feature grids rather than
        computing new ones.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A QueryFeatures object which contains the required features for every
        required resolution.
    """

    if feature_grids is None:
      feature_grids = self.get_feature_grids(
          video,
          is_training=is_training,
          refinement_resolutions=refinement_resolutions,
      )

    feature_grid = feature_grids.lowres
    hires_feats = feature_grids.hires
    resize_im_shape = feature_grids.resolutions

    shape = video.shape
    # shape is [batch_size, time, height, width, channels]; conversion needs
    # [time, width, height]
    curr_resolution = (-1, -1)
    query_feats = []
    hires_query_feats = []
    for i, resolution in enumerate(resize_im_shape):
      if utils.is_same_res(curr_resolution, resolution):
        query_feats.append(query_feats[-1])
        hires_query_feats.append(hires_query_feats[-1])
        continue
      position_in_grid = utils.convert_grid_coordinates(
          query_points,
          shape[1:4],
          feature_grid[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_hires = utils.convert_grid_coordinates(
          query_points,
          shape[1:4],
          hires_feats[i].shape[1:4],
          coordinate_format='tyx',
      )

      interp_features = utils.map_coordinates_3d(
          feature_grid[i], position_in_grid
      )
      hires_interp = utils.map_coordinates_3d(
          hires_feats[i], position_in_grid_hires
      )

      hires_query_feats.append(hires_interp)
      query_feats.append(interp_features)

    return QueryFeatures(
        tuple(query_feats), tuple(hires_query_feats), tuple(resize_im_shape)
    )

  def get_feature_grids(
      self,
      video: torch.Tensor,
      is_training: bool,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> FeatureGrids:
    """Computes feature grids.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      refinement_resolutions: A list of (height, width) tuples. Refinement will
        be repeated at each specified resolution, to achieve high accuracy on
        resolutions higher than what TAPIR was trained on. If None, reasonable
        refinement resolutions will be inferred from the input video size.

    Returns:
      A FeatureGrids object containing the required features for every
      required resolution. Note that there will be one more feature grid
      than there are refinement_resolutions, because there is always a
      feature grid computed for TAP-Net initialization.
    """
    del is_training
    if refinement_resolutions is None:
      refinement_resolutions = utils.generate_default_resolutions(
          video.shape[2:4], self.initial_resolution
      )

    all_required_resolutions = [self.initial_resolution]
    all_required_resolutions.extend(refinement_resolutions)

    feature_grid = []
    hires_feats = []
    resize_im_shape = []
    curr_resolution = (-1, -1)

    latent = None
    hires = None
    video_resize = None
    for resolution in all_required_resolutions:
      if resolution[0] % 8 != 0 or resolution[1] % 8 != 0:
        raise ValueError('Image resolution must be a multiple of 8.')

      if not utils.is_same_res(curr_resolution, resolution):
        if utils.is_same_res(curr_resolution, video.shape[-3:-1]):
          video_resize = video
        else:
          video_resize = utils.bilinear(video, resolution)

        curr_resolution = resolution
        n, f, h, w, c = video_resize.shape
        video_resize = video_resize.view(n*f, h, w, c).permute(0, 3, 1, 2)

        if self.feature_extractor_chunk_size > 0:
          latent_list = []
          hires_list = []
          chunk_size = self.feature_extractor_chunk_size
          for start_idx in range(0, video_resize.shape[0], chunk_size):
            video_chunk = video_resize[start_idx:start_idx + chunk_size]
            resnet_out = self.resnet_torch(video_chunk)

            u3 = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
            latent_list.append(u3)
            u1 = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)
            hires_list.append(u1)

          latent = torch.cat(latent_list, dim=0)
          hires = torch.cat(hires_list, dim=0)

        else:
          resnet_out = self.resnet_torch(video_resize)
          latent = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
          hires = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)

        if self.extra_convs:
          latent = self.extra_convs(latent)

        latent = latent / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(latent), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=latent.device),
            )
        )
        hires = hires / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(hires), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=hires.device),
            )
        )

        latent = latent.view(n, f, *latent.shape[1:])
        hires = hires.view(n, f, *hires.shape[1:])

      feature_grid.append(latent)
      hires_feats.append(hires)
      resize_im_shape.append(video_resize.shape[2:4])

    return FeatureGrids(
        tuple(feature_grid), tuple(hires_feats), tuple(resize_im_shape)
    )

  def estimate_trajectories(
      self,
      video_size: Tuple[int, int],
      is_training: bool,
      feature_grids: FeatureGrids,
      query_features: QueryFeatures,
      query_points_in_video: Optional[torch.Tensor],
      query_chunk_size: Optional[int] = None,
      causal_context: Optional[dict[str, torch.Tensor]] = None,
      get_causal_context: bool = False,
  ) -> Mapping[str, Any]:
    """Estimates trajectories given features for a video and query features.

    Args:
      video_size: A 2-tuple containing the original [height, width] of the
        video.  Predictions will be scaled with respect to this resolution.
      is_training: Whether we are training.
      feature_grids: a FeatureGrids object computed for the given video.
      query_features: a QueryFeatures object computed for the query points.
      query_points_in_video: If provided, assume that the query points come from
        the same video as feature_grids, and therefore constrain the resulting
        trajectories to (approximately) pass through them.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      causal_context: If provided, a dict of causal context to use for
        refinement.
      get_causal_context: If True, return causal context in the output.

    Returns:
      A dict of outputs, including:
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          in raster coordinates
        expected_dist: uncertainty estimate logits, of shape
          [batch, num_queries, num_frames], where higher indicates more likely
          to be far from the correct answer.
    """
    del is_training

    def train2orig(x):
      return utils.convert_grid_coordinates(
          x,
          self.initial_resolution[::-1],
          video_size[::-1],
          coordinate_format='xy',
      )

    occ_iters = []
    pts_iters = []
    expd_iters = []
    new_causal_context = []
    num_iters = self.num_pips_iter * (len(feature_grids.lowres) - 1)
    for _ in range(num_iters + 1):
      occ_iters.append([])
      pts_iters.append([])
      expd_iters.append([])
      new_causal_context.append([])
    del new_causal_context[-1]

    infer = functools.partial(
        self.tracks_from_cost_volume,
        im_shp=feature_grids.lowres[0].shape[0:2]
        + self.initial_resolution
        + (3,),
    )

    num_queries = query_features.lowres[0].shape[1]
    if causal_context is None:
      perm = torch.randperm(num_queries)
    else:
      perm = torch.arange(num_queries)

    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(num_queries)

    for ch in range(0, num_queries, query_chunk_size):
      perm_chunk = perm[ch : ch + query_chunk_size]
      chunk = query_features.lowres[0][:, perm_chunk]

      cc_chunk = []
      if causal_context is not None:
        for d in range(len(causal_context)):
          tmp_dict = {}
          for k, v in causal_context[d].items():
            tmp_dict[k] = v[:, perm_chunk]
          cc_chunk.append(tmp_dict)

      if query_points_in_video is not None:
        infer_query_points = query_points_in_video[
            :, perm[ch : ch + query_chunk_size]
        ]
        num_frames = feature_grids.lowres[0].shape[1]
        infer_query_points = utils.convert_grid_coordinates(
            infer_query_points,
            (num_frames,) + video_size,
            (num_frames,) + self.initial_resolution,
            coordinate_format='tyx',
        )
      else:
        infer_query_points = None

      points, occlusion, expected_dist = infer(
          chunk,
          feature_grids.lowres[0],
          infer_query_points,
      )
      pts_iters[0].append(train2orig(points))
      occ_iters[0].append(occlusion)
      expd_iters[0].append(expected_dist)

      mixer_feats = None
      for i in range(num_iters):
        feature_level = i // self.num_pips_iter + 1
        queries = [
            query_features.hires[feature_level][:, perm_chunk],
            query_features.lowres[feature_level][:, perm_chunk],
        ]
        for _ in range(self.pyramid_level):
          queries.append(queries[-1])
        pyramid = [
            feature_grids.hires[feature_level],
            feature_grids.lowres[feature_level],
        ]
        for _ in range(self.pyramid_level):
          pyramid.append(
              F.avg_pool3d(
                  pyramid[-1],
                  kernel_size=(2, 2, 1),
                  stride=(2, 2, 1),
                  padding=0,
              )
          )
        cc = cc_chunk[i] if causal_context is not None else None
        refined = self.refine_pips(
            queries,
            None,
            pyramid,
            points,
            occlusion,
            expected_dist,
            orig_hw=self.initial_resolution,
            last_iter=mixer_feats,
            mixer_iter=i,
            resize_hw=feature_grids.resolutions[feature_level],
            causal_context=cc,
            get_causal_context=get_causal_context,
        )
        points, occlusion, expected_dist, mixer_feats, new_causal = refined
        pts_iters[i + 1].append(train2orig(points))
        occ_iters[i + 1].append(occlusion)
        expd_iters[i + 1].append(expected_dist)
        new_causal_context[i].append(new_causal)

        if (i + 1) % self.num_pips_iter == 0:
          mixer_feats = None
          expected_dist = expd_iters[0][-1]
          occlusion = occ_iters[0][-1]

    occlusion = []
    points = []
    expd = []
    for i, _ in enumerate(occ_iters):
      occlusion.append(torch.cat(occ_iters[i], dim=1)[:, inv_perm])
      points.append(torch.cat(pts_iters[i], dim=1)[:, inv_perm])
      expd.append(torch.cat(expd_iters[i], dim=1)[:, inv_perm])

    for i in range(len(new_causal_context)):
      combined_dict = {}
      for key in new_causal_context[i][0].keys():
        arrays = [d[key] for d in new_causal_context[i]]
        concatenated = torch.cat(arrays, dim=1)
        combined_dict[key] = concatenated[:, inv_perm]
      new_causal_context[i] = combined_dict

    out = dict(
        occlusion=occlusion,
        tracks=points,
        expected_dist=expd,
    )

    if get_causal_context:
      out['causal_context'] = new_causal_context
    return out

  def refine_pips(
      self,
      target_feature,
      frame_features,
      pyramid,
      pos_guess,
      occ_guess,
      expd_guess,
      orig_hw,
      last_iter=None,
      mixer_iter=0.0,
      resize_hw=None,
      causal_context=None,
      get_causal_context=False,
  ):
    del frame_features
    del mixer_iter
    orig_h, orig_w = orig_hw
    resized_h, resized_w = resize_hw
    corrs_pyr = []
    assert len(target_feature) == len(pyramid)
    for pyridx, (query, grid) in enumerate(zip(target_feature, pyramid)):
      # note: interp needs [y,x]
      coords = utils.convert_grid_coordinates(
          pos_guess, (orig_w, orig_h), grid.shape[-2:-4:-1]
      )
      coords = torch.flip(coords, dims=(-1,))
      last_iter_query = None
      if last_iter is not None:
        if pyridx == 0:
          last_iter_query = last_iter[..., : self.highres_dim]
        else:
          last_iter_query = last_iter[..., self.highres_dim :]

      ctxy, ctxx = torch.meshgrid(
          torch.arange(-3, 4), torch.arange(-3, 4), indexing='ij'
      )
      ctx = torch.stack([ctxy, ctxx], dim=-1)
      ctx = ctx.reshape(-1, 2).to(coords.device)
      coords2 = coords.unsqueeze(3) + ctx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
      neighborhood = utils.map_coordinates_2d(grid, coords2)

      # s is spatial context size
      if last_iter_query is None:
        patches = torch.einsum('bnfsc,bnc->bnfs', neighborhood, query)
      else:
        patches = torch.einsum(
            'bnfsc,bnfc->bnfs', neighborhood, last_iter_query
        )

      corrs_pyr.append(patches)
    corrs_pyr = torch.concatenate(corrs_pyr, dim=-1)

    corrs_chunked = corrs_pyr
    pos_guess_input = pos_guess
    occ_guess_input = occ_guess[..., None]
    expd_guess_input = expd_guess[..., None]

    # mlp_input is batch, num_points, num_chunks, frames_per_chunk, channels
    if last_iter is None:
      both_feature = torch.cat([target_feature[0], target_feature[1]], axis=-1)
      mlp_input_features = torch.tile(
          both_feature.unsqueeze(2), (1, 1, corrs_chunked.shape[-2], 1)
      )
    else:
      mlp_input_features = last_iter

    pos_guess_input = torch.zeros_like(pos_guess_input)

    mlp_input = torch.cat(
        [
            pos_guess_input,
            occ_guess_input,
            expd_guess_input,
            mlp_input_features,
            corrs_chunked,
        ],
        axis=-1,
    )
    x = utils.einshape('bnfc->(bn)fc', mlp_input)

    if causal_context is not None:
      for k, v in causal_context.items():
        causal_context[k] = utils.einshape('bn...->(bn)...', v)
    res, new_causal_context = self.torch_pips_mixer(
        x.float(), causal_context, get_causal_context
    )
    res = utils.einshape('(bn)fc->bnfc', res, b=mlp_input.shape[0])
    if get_causal_context:
      for k, v in new_causal_context.items():
        new_causal_context[k] = utils.einshape(
            '(bn)...->bn...', v, b=mlp_input.shape[0]
        )

    pos_update = utils.convert_grid_coordinates(
        res[..., :2],
        (resized_w, resized_h),
        (orig_w, orig_h),
    )
    return (
        pos_update + pos_guess,
        res[..., 2] + occ_guess,
        res[..., 3] + expd_guess,
        res[..., 4:] + (mlp_input_features if last_iter is None else last_iter),
        new_causal_context,
    )

  def tracks_from_cost_volume(
      self,
      interp_feature: torch.Tensor,
      feature_grid: torch.Tensor,
      query_points: Optional[torch.Tensor],
      im_shp=None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts features into tracks by computing a cost volume.

    The computed cost volume will have shape
      [batch, num_queries, time, height, width], which can be very
      memory intensive.

    Args:
      interp_feature: A tensor of features for each query point, of shape
        [batch, num_queries, channels, heads].
      feature_grid: A tensor of features for the video, of shape [batch, time,
        height, width, channels, heads].
      query_points: When computing tracks, we assume these points are given as
        ground truth and we reproduce them exactly.  This is a set of points of
        shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
        raster coordinates.
      im_shp: The shape of the original image, i.e., [batch, num_frames, time,
        height, width, 3].

    Returns:
      A 2-tuple of the inferred points (of shape
        [batch, num_points, num_frames, 2] where each point is [x, y]) and
        inferred occlusion (of shape [batch, num_points, num_frames], where
        each is a logit where higher means occluded)
    """

    mods = self.torch_cost_volume_track_mods
    cost_volume = torch.einsum(
        'bnc,bthwc->tbnhw',
        interp_feature,
        feature_grid,
    )

    shape = cost_volume.shape
    batch_size, num_points = cost_volume.shape[1:3]
    cost_volume = utils.einshape('tbnhw->(tbn)hw1', cost_volume)

    cost_volume = cost_volume.permute(0, 3, 1, 2)
    occlusion = mods['hid1'](cost_volume)
    occlusion = torch.nn.functional.relu(occlusion)

    pos = mods['hid2'](occlusion)
    pos = pos.permute(0, 2, 3, 1)
    pos_rshp = utils.einshape('(tb)hw1->t(b)hw1', pos, t=shape[0])

    pos = utils.einshape(
        't(bn)hw1->bnthw', pos_rshp, b=batch_size, n=num_points
    )
    pos_sm = pos.reshape(pos.size(0), pos.size(1), pos.size(2), -1)
    softmaxed = F.softmax(pos_sm * self.softmax_temperature, dim=-1)
    pos = softmaxed.view_as(pos)

    points = utils.heatmaps_to_points(pos, im_shp, query_points=query_points)

    occlusion = torch.nn.functional.pad(occlusion, (0, 2, 0, 2))
    occlusion = mods['hid3'](occlusion)
    occlusion = torch.nn.functional.relu(occlusion)
    occlusion = torch.mean(occlusion, dim=(-1, -2))
    occlusion = mods['hid4'](occlusion)
    occlusion = torch.nn.functional.relu(occlusion)
    occlusion = mods['occ_out'](occlusion)

    expected_dist = utils.einshape(
        '(tbn)1->bnt', occlusion[..., 1:2], n=shape[2], t=shape[0]
    )
    occlusion = utils.einshape(
        '(tbn)1->bnt', occlusion[..., 0:1], n=shape[2], t=shape[0]
    )
    return points, occlusion, expected_dist

  def construct_initial_causal_state(self, num_points, num_resolutions=1):
    """Construct initial causal state."""
    value_shapes = {}
    for i in range(self.num_mixer_blocks):
      value_shapes[f'block_{i}_causal_1'] = (1, num_points, 2, 512)
      value_shapes[f'block_{i}_causal_2'] = (1, num_points, 2, 2048)
    fake_ret = {
        k: torch.zeros(v, dtype=torch.float32) for k, v in value_shapes.items()
    }
    return [fake_ret] * num_resolutions * 4

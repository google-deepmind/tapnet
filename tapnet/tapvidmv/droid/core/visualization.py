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

"""Interactive 3D point cloud and 2D/4D video visualization utilities."""

# pylint: disable=invalid-name

import core.geometry
import cv2
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm


def occlusion_colors(is_robot, visible):
  palette = np.array(
      [[34, 220, 100], [255, 65, 65], [56, 189, 248], [250, 204, 21]],
      dtype=np.uint8,
  )
  return palette[
      2 * np.asarray(is_robot, dtype=int) + ~np.asarray(visible, dtype=bool)
  ]


def draw_label(img, text, org, scale, colour, thickness, outline):
  cv2.putText(
      img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline
  )
  cv2.putText(
      img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness
  )


def inspect_dict_structure(data, name="episode", indent=0):
  """Prints a recursive summary of nested episode dictionaries and arrays."""
  prefix = "  " * indent
  if isinstance(data, dict):
    print(f"{prefix}{name} (dict, {len(data)} keys)")
    for key, value in data.items():
      inspect_dict_structure(value, name=str(key), indent=indent + 1)
  elif getattr(data, "ndim", 0) > 0:
    print(
        f"{prefix}{name}: {type(data).__name__}, shape={tuple(data.shape)},"
        f" dtype={data.dtype}"
    )
  elif isinstance(data, (list, tuple)):
    print(f"{prefix}{name}: {type(data).__name__}, len={len(data)}")
  else:
    print(f"{prefix}{name}: {type(data).__name__} = {str(data)[:50]}")


def fuse_cameras(episode, poses, t, device="cpu", max_depth=None):
  """Unprojects and merges RGB-D point clouds across all cameras at frame t."""
  clouds = []
  for cam_id, cam in sorted(episode["camera"].items()):
    depth = cam["raw_depth"][t]
    if max_depth is not None:
      depth = np.where(depth < max_depth, depth, 0)
    clouds.append(
        core.geometry.unproject_depth_torch(
            depth,
            cam["video_rgb"][t],
            cam["K"],
            poses[cam_id]["T_cam2world"][t],
            device,
        )
    )
  points, colors = zip(*clouds, strict=True)
  return torch.cat(points), torch.cat(colors)


def show_point_cloud(
    points,
    colors,
    title,
    eye,
    up=(0, 0, 1),
    size=1.5,
    max_points=150000,
    height=600,
    width=1000,
):
  """Displays an interactive 3D RGB point cloud using Plotly."""
  idx = np.random.permutation(len(points))[:max_points]
  points, colors = points[idx], colors[idx]
  go.Figure(
      data=[
          go.Scatter3d(
              x=points[:, 0],
              y=points[:, 1],
              z=points[:, 2],
              mode="markers",
              marker=dict(
                  size=size, color=[f"rgb({r},{g},{b})" for r, g, b in colors]
              ),
          )
      ],
      layout=go.Layout(
          title=title,
          margin=dict(l=0, r=0, b=0, t=40),
          width=width,
          height=height,
          showlegend=False,
          scene=dict(
              aspectmode="data",
              camera=dict(eye=dict(zip("xyz", eye)), up=dict(zip("xyz", up))),
          ),
      ),
  ).show()


def show_fused_point_cloud(
    episode, poses, t=0, max_depth=None, height=600, width=1000
):
  points, colors = fuse_cameras(episode, poses, t, max_depth=max_depth)
  show_point_cloud(
      points.numpy(),
      colors.numpy(),
      title=f"Fused Point Cloud (Frame {t})",
      eye=(-1.2, -1.2, 0.8),
      height=height,
      width=width,
  )


def show_distilled_gripper_3d(median_depth, K, img_rgb):
  v, u = np.where(median_depth > 0)
  points = core.geometry.unproject_camera_frame(u, v, median_depth[v, u], K)[
      :3
  ].T
  show_point_cloud(
      points,
      img_rgb[v, u],
      title="Distilled Gripper Surface",
      eye=(0, -0.5, -1.5),
      up=(0, -1, 0),
      size=2,
  )


def show_gripper_refinement(episode, t=0):
  """Plots wrist RGB+SAM mask alongside raw, distilled, and refined depth."""
  wrist_cam_id = episode["meta"]["wrist_cam_id"]
  cam_data = episode["camera"][wrist_cam_id]
  rgb = cam_data["video_rgb"][t]

  overlay = rgb.copy()
  overlay[cam_data["sam_real_masks"][t]] = [255, 0, 128]

  panels = [
      (cam_data["original_raw_depth"][t], "Original Sensor Depth"),
      (cam_data["empirical_gripper_depth"], "Distilled Gripper Surface Depth"),
      (cam_data["raw_depth"][t], "Final Refined Depth (Injected)"),
  ]

  _, axes = plt.subplots(1, 1 + len(panels), figsize=(20, 4.5))
  axes[0].imshow(cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0))
  axes[0].set_title("RGB + SAM Gripper Mask", fontsize=11)
  for ax, (depth, title) in zip(axes[1:], panels, strict=True):
    im = ax.imshow(
        np.where(depth > 0, depth, np.nan), cmap="viridis", vmin=0.1, vmax=1.2
    )
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
  for ax in axes:
    ax.axis("off")

  plt.suptitle(
      f"Wrist Camera [{wrist_cam_id[:8]}] Gripper Refinement Inspection (Frame"
      f" {t})",
      fontsize=13,
      y=1.02,
  )
  plt.tight_layout()
  plt.show()


def colorize_disparity(disparity, vmax=100.0):
  levels = (np.clip(disparity, 0, vmax) / vmax * 255).astype(np.uint8)
  return np.stack([
      cv2.cvtColor(cv2.applyColorMap(f, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
      for f in levels
  ])


def render_multicam_disparity_video(
    episode, max_frames=None, tgt_size=(128, 228)
):
  """Renders side-by-side stereo RGB and colorized disparity across cameras."""
  rows = []
  for cam_data in episode["camera"].values():
    depth = cam_data["raw_depth"][:max_frames]
    disparity = np.divide(
        cam_data["K"][0, 0] * cam_data["baseline"],
        depth,
        out=np.zeros_like(depth),
        where=depth > 0,
    )
    rows.append(
        np.concatenate(
            [
                media.resize_video(
                    cam_data["video_rgb"][:max_frames], tgt_size
                ),
                media.resize_video(
                    cam_data["video_right"][:max_frames], tgt_size
                ),
                colorize_disparity(media.resize_video(disparity, tgt_size)),
            ],
            axis=2,
        )
    )
  return np.concatenate(rows, axis=1)


def render_2d_tracking_video(
    video_frames,
    tracks,
    visibility,
    global_colors=None,
    linewidth=3,
    tracks_leave_trace=20,
    tgt_size=None,
    max_frames=None,
):
  """Renders 2D point tracks with motion trails and occlusion rings on video."""
  video_frames, tracks, visibility = (
      video_frames[:max_frames],
      tracks[:max_frames],
      visibility[:max_frames],
  )

  if tgt_size is not None:
    src_height, src_width = video_frames[0].shape[:2]
    video_frames = media.resize_video(np.array(video_frames), tgt_size)
    tracks = tracks * [tgt_size[1] / src_width, tgt_size[0] / src_height]

  if global_colors is None:
    depth_order = tracks[0, :, 1]
    global_colors = (
        plt.cm.gist_rainbow(
            plt.Normalize(depth_order.min(), depth_order.max())(depth_order)
        )[:, :3]
        * 255
    )
  colors = [tuple(map(int, c)) for c in global_colors]

  height, width = video_frames[0].shape[:2]
  pts = np.round(tracks).astype(np.int32)
  in_frame = core.geometry.in_frame(
      tracks[..., 0], tracks[..., 1], width, height
  )
  drawable = in_frame & visibility

  radius = int(linewidth * 2)
  video_frames = [frame.copy() for frame in video_frames]
  for t, img in enumerate(video_frames):
    trace = min(t, tracks_leave_trace)
    for step in range(trace):
      past = t - trace + step
      alpha = (step / (trace + 1)) ** 2
      overlay = img.copy()
      for i in np.flatnonzero(drawable[past] & drawable[past + 1]):
        cv2.line(
            overlay,
            tuple(pts[past, i]),
            tuple(pts[past + 1, i]),
            colors[i],
            linewidth,
            cv2.LINE_AA,
        )
      cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for i in np.flatnonzero(drawable[t]):
      cv2.circle(img, tuple(pts[t, i]), radius, colors[i], -1, cv2.LINE_AA)

    occluded = img.copy()
    for i in np.flatnonzero(in_frame[t] & ~visibility[t]):
      cv2.circle(occluded, tuple(pts[t, i]), radius, colors[i], 1, cv2.LINE_AA)
    cv2.addWeighted(occluded, 0.35, img, 0.65, 0, img)

  return video_frames


def render_segmentation_video(
    episode, poses, pb_renderer, tgt_width=1200, max_frames=None
):
  """Overlays rendered URDF robot masks onto multi-camera RGB video frames."""
  robot = episode["robot"]
  n_frames = len(robot["joint_positions"][:max_frames])

  video_frames = []
  for t in tqdm(range(n_frames), desc="Rendering segmentation"):
    pb_renderer.update_robot_pose(
        robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t]
    )
    views = []
    for cam_id, cam_data in episode["camera"].items():
      img_rgb = cam_data["video_rgb"][t]
      height, width = img_rgb.shape[:2]
      robot_mask = (
          pb_renderer.render_depth(
              poses[cam_id]["T_cam2world"][t], cam_data["K"], width, height
          )
          > 0
      )
      overlay = img_rgb.copy()
      overlay[robot_mask] = [50, 150, 255]
      blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
      draw_label(
          blended, f"Cam [{cam_id}]", (20, 50), 1.2, (255, 255, 255), 2, 4
      )
      views.append(blended)
    row = np.concatenate(views, axis=1)
    video_frames.append(
        cv2.resize(
            row, (tgt_width, round(row.shape[0] * tgt_width / row.shape[1]))
        )
    )
  return video_frames


def render_cross_camera_axes(
    episode, poses, max_frames=None, tgt_width=1200, axis_len=0.15
):
  """Projects and draws 3D coordinate axes of other cameras into each view."""
  cam_ids = list(episode["camera"])
  n_frames = len(poses[cam_ids[0]]["T_cam2world"][:max_frames])
  axes_3d = np.array([
      [0, 0, 0, 1],
      [axis_len, 0, 0, 1],
      [0, axis_len, 0, 1],
      [0, 0, axis_len, 1],
  ]).T

  video_frames = []
  for t in tqdm(range(n_frames), desc="Rendering camera axes"):
    views = []
    for obs_cam in cam_ids:
      cam_data = episode["camera"][obs_cam]
      img_rgb = cam_data["video_rgb"][t].copy()
      height, width = img_rgb.shape[:2]
      T_world2obs = np.linalg.inv(poses[obs_cam]["T_cam2world"][t])

      for tgt_cam in cam_ids:
        if tgt_cam == obs_cam:
          continue
        points_cam = (T_world2obs @ poses[tgt_cam]["T_cam2world"][t] @ axes_3d)[
            :3, :
        ]
        if points_cam[2, 0] < 0:
          continue
        uv = cam_data["K"] @ points_cam
        org, px, py, pz = map(tuple, (uv[:2] / uv[2]).astype(int).T)
        if not (0 <= org[0] < width and 0 <= org[1] < height):
          continue
        for tip, colour in zip(
            (px, py, pz), ((255, 0, 0), (0, 255, 0), (0, 0, 255)), strict=True
        ):
          cv2.line(img_rgb, org, tip, colour, 3)
        cv2.circle(img_rgb, org, 5, (0, 0, 0), -1)
        cv2.circle(img_rgb, org, 2, (255, 255, 255), -1)
        draw_label(
            img_rgb,
            f"Cam {tgt_cam}",
            (org[0] + 8, org[1] - 8),
            0.6,
            (255, 255, 255),
            2,
            3,
        )

      draw_label(
          img_rgb, f"View: {obs_cam}", (15, 35), 0.8, (0, 255, 255), 2, 3
      )
      views.append(img_rgb)
    row = np.concatenate(views, axis=1)
    video_frames.append(
        cv2.resize(
            row, (tgt_width, round(row.shape[0] * tgt_width / row.shape[1]))
        )
    )
  return video_frames


def disc_offsets(radius, device):
  span = torch.arange(-radius, radius + 1, device=device)
  du, dv = (
      offsets.reshape(-1)
      for offsets in torch.meshgrid(span, span, indexing="ij")
  )
  on_disc = du**2 + dv**2 <= radius**2
  return du[on_disc], dv[on_disc]


def splat(points, colors, K, T_cam2world, height, width, radii):
  """Renders colored 3D discs with z-buffering onto an image tensor."""
  T_world2cam = torch.linalg.inv(T_cam2world)
  points_cam = points @ T_world2cam[:3, :3].T + T_world2cam[:3, 3]
  z_cam = points_cam[:, 2]
  uv = (points_cam @ K.T)[:, :2] / z_cam[:, None].clamp(min=1e-6)
  u_cam, v_cam = uv.round().long().unbind(-1)

  hits = []
  for radius in radii.unique():
    on = radii == radius
    du, dv = disc_offsets(int(radius), points.device)
    u, v = u_cam[on][:, None] + du, v_cam[on][:, None] + dv
    z = z_cam[on][:, None].expand_as(u)
    keep = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    hits.append((
        v[keep] * width + u[keep],
        z[keep],
        colors[on][:, None, :].expand(-1, len(du), -1)[keep],
    ))
  idx, z, colors = (torch.cat(part) for part in zip(*hits, strict=True))

  depth = torch.full((height * width,), torch.inf, device=points.device)
  depth.scatter_reduce_(0, idx, z, reduce="amin", include_self=False)

  img = torch.zeros(
      (height * width, 3), dtype=torch.uint8, device=points.device
  )
  wins = z == depth[idx]
  img[idx[wins]] = colors[wins]
  return img.reshape(height, width, 3)


def as_tensor(array, dtype, device):
  if torch.is_tensor(array):
    return array.to(dtype=dtype, device=device)
  return torch.as_tensor(
      np.ascontiguousarray(array), dtype=dtype, device=device
  )


def point_layer(points, colors, radius, device):
  points = as_tensor(points, torch.float32, device)
  radii = torch.full((len(points),), radius, dtype=torch.long, device=device)
  return points, as_tensor(colors, torch.uint8, device), radii


def line_layer(starts, ends, colors, radius, device, samples=64):
  starts, ends = as_tensor(starts, torch.float32, device), as_tensor(
      ends, torch.float32, device
  )
  alpha = torch.linspace(0, 1, samples, device=device)[None, :, None]
  points = (starts[:, None] + (ends - starts)[:, None] * alpha).reshape(-1, 3)
  return point_layer(
      points,
      as_tensor(colors, torch.uint8, device).repeat_interleave(samples, 0),
      radius,
      device,
  )


def look_at(eye, target, up=(0, 0, 1)):
  forward = np.subtract(target, eye, dtype=float)
  forward /= np.linalg.norm(forward) + 1e-6
  right = np.cross(forward, up)
  right /= np.linalg.norm(right) + 1e-6
  pose = np.eye(4)
  pose[:3, :4] = np.column_stack(
      [right, np.cross(forward, right), forward, eye]
  )
  return pose


def frustum_wireframe(K_aspect, depth, fov_y):
  half_h = depth * np.tan(np.radians(fov_y / 2))
  half_w = half_h * K_aspect
  corners = np.array([
      [0, 0, 0],
      [-half_w, -half_h, depth],
      [half_w, -half_h, depth],
      [half_w, half_h, depth],
      [-half_w, half_h, depth],
  ])
  return corners, np.array(
      [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
  )


def render_4d_orbit_with_tracks(
    episode,
    poses,
    tracks_3d=None,
    max_frames=None,
    width=640,
    height=360,
    fov_y=60.0,
    orbit_center=(0.4, 0.0, 0.0),
    orbit_radius=1.2,
    camera_height=0.5,
    angle_start=np.pi / 2,
    max_depth=None,
    max_render_points=400000,
    max_render_tracks=500,
    track_history=5,
    point_size=1,
    track_size=3,
    frustum_depth=0.15,
    frustum_aspect=4.0 / 3.0,
):
  """Renders an orbiting 3D point-cloud video with camera frustums and tracks."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  cam_ids = sorted(episode["camera"])
  n_frames = len(poses[cam_ids[0]]["T_cam2world"][:max_frames])

  focal = (height / 2) / np.tan(np.radians(fov_y) / 2)
  K_viz = torch.tensor(
      [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
      dtype=torch.float32,
      device=device,
  )

  if tracks_3d is not None:
    if tracks_3d.shape[1] > max_render_tracks:
      tracks_3d = tracks_3d[
          :, np.random.permutation(tracks_3d.shape[1])[:max_render_tracks]
      ]
    depth_order = tracks_3d[0, :, 1]
    norm = plt.Normalize(depth_order.min(), depth_order.max())
    track_colors = (plt.cm.hsv(norm(depth_order))[:, :3] * 255).astype(np.uint8)

  corners_cam, frustum_edges = frustum_wireframe(
      frustum_aspect, frustum_depth, fov_y
  )
  cam_colors = np.array(
      [[255, 102, 51], [51, 204, 51], [51, 102, 255], [255, 255, 51]],
      dtype=np.uint8,
  )
  cam_colors = cam_colors[np.arange(len(cam_ids)) % len(cam_colors)]

  video_frames = []
  for t in tqdm(range(n_frames), desc="Rendering 4D orbit"):
    cloud, cloud_colors = fuse_cameras(episode, poses, t, device, max_depth)
    if len(cloud) > max_render_points:
      keep = torch.randperm(len(cloud), device=device)[:max_render_points]
      cloud, cloud_colors = cloud[keep], cloud_colors[keep]
    layers = [point_layer(cloud, cloud_colors, point_size, device)]

    if tracks_3d is not None:
      layers.append(point_layer(tracks_3d[t], track_colors, track_size, device))

      trail = tracks_3d[max(0, t - track_history) : t + 1]
      starts, ends = trail[:-1].reshape(-1, 3), trail[1:].reshape(-1, 3)
      moved = np.linalg.norm(ends - starts, axis=1) > 1e-6
      if moved.any():
        trail_colors = np.tile(track_colors, (len(trail) - 1, 1))
        layers.append(
            line_layer(
                starts[moved],
                ends[moved],
                trail_colors[moved],
                max(track_size - 2, 0),
                device,
            )
        )

    extrinsics = np.stack(
        [poses[cam_id]["T_cam2world"][t] for cam_id in cam_ids]
    )
    corners = (
        corners_cam @ extrinsics[:, :3, :3].transpose(0, 2, 1)
        + extrinsics[:, None, :3, 3]
    )
    layers.append(
        line_layer(
            corners[:, frustum_edges[:, 0]].reshape(-1, 3),
            corners[:, frustum_edges[:, 1]].reshape(-1, 3),
            np.repeat(cam_colors, len(frustum_edges), axis=0),
            0,
            device,
        )
    )

    angle = angle_start + t * np.pi / n_frames
    eye = np.array(
        [np.cos(angle) * orbit_radius, np.sin(angle) * orbit_radius, 0]
    ) + [*orbit_center[:2], camera_height]
    viz_pose = as_tensor(look_at(eye, orbit_center), torch.float32, device)

    points, colors, radii = (
        torch.cat(parts) for parts in zip(*layers, strict=True)
    )
    img_rgb = (
        splat(points, colors, K_viz, viz_pose, height, width, radii)
        .cpu()
        .numpy()
    )
    draw_label(img_rgb, f"Frame: {t:03d}", (30, 50), 0.7, (255, 255, 255), 2, 4)
    video_frames.append(img_rgb)

  return video_frames

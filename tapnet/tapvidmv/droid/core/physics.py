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

"""PyBullet URDF forward kinematics and offscreen depth/segmentation rendering."""

# pylint: disable=invalid-name

import importlib.util
import multiprocessing
import os

import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

NEAR_PLANE = 0.01
FAR_PLANE = 10.0
GRIPPER_ANGLE_SCALE = 0.8028
GRIPPER_WIDTH_OFFSET = 0.08


def metric_depth(depth_buf, height, width):
  buf = np.reshape(depth_buf, (height, width)).astype(np.float32)
  metric = FAR_PLANE * NEAR_PLANE / (FAR_PLANE - buf * (FAR_PLANE - NEAR_PLANE))
  return np.where(metric < FAR_PLANE * 0.99, metric, 0.0)


def link_transform(obj_id, link_id):
  if link_id == -1:
    pos, orn = pybullet.getBasePositionAndOrientation(obj_id)
  else:
    pos, orn = pybullet.getLinkState(obj_id, link_id)[:2]

  T_link2world = np.eye(4, dtype=np.float32)
  T_link2world[:3, :3] = Rotation.from_quat(orn).as_matrix()
  T_link2world[:3, 3] = pos
  return T_link2world


def _load_egl():
  cuda_pin = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
  if "EGL_VISIBLE_DEVICES" not in os.environ and cuda_pin.isdigit():
    os.environ["EGL_VISIBLE_DEVICES"] = cuda_pin

  spec = importlib.util.find_spec("eglRenderer")
  if spec is None:
    return False
  return pybullet.loadPlugin(spec.origin, "_eglRendererPlugin") >= 0


class PyBulletRenderer:
  """Offscreen PyBullet renderer for the Franka + Robotiq 2F-85 robot arm."""

  def __init__(self, urdf, gpu):
    if pybullet.isConnected():
      pybullet.disconnect()
    pybullet.connect(pybullet.DIRECT)

    if not pybullet.isNumpyEnabled():
      raise RuntimeError(
          "pybullet was built without NumPy: getCameraImage is 2x slower. Rerun"
          " bash setup.sh"
      )

    self.urdf = urdf
    self.gpu = bool(gpu) and _load_egl()
    if gpu and not self.gpu:
      raise RuntimeError(
          "EGL requested but the plugin did not load. --config.render.gpu=False"
      )
    self.render_mode = (
        pybullet.ER_BULLET_HARDWARE_OPENGL
        if self.gpu
        else pybullet.ER_TINY_RENDERER
    )

    self.robot_id = pybullet.loadURDF(
        urdf, useFixedBase=True, flags=pybullet.URDF_IGNORE_COLLISION_SHAPES
    )

    self.arm_joints = []
    self.gripper_joints = []
    self.gripper_signs = []
    self.gripper_links = []
    for i in range(pybullet.getNumJoints(self.robot_id)):
      info = pybullet.getJointInfo(self.robot_id, i)
      joint_name = info[1].decode()
      if "panda_link" not in info[12].decode():
        self.gripper_links.append(i)
      if info[2] == pybullet.JOINT_FIXED:
        continue
      if "panda_joint" in joint_name:
        self.arm_joints.append(i)
        continue
      self.gripper_joints.append(i)
      sign = -1 if "right" in joint_name else 1
      if any(
          k in joint_name for k in ["inner_finger", "follower", "finger_tip"]
      ):
        sign = -sign
      self.gripper_signs.append(sign)

  def update_robot_pose(self, joint_angles, gripper_state):
    """Updates arm and mimicked gripper joint angles in PyBullet."""
    for i, angle in zip(self.arm_joints, joint_angles):
      pybullet.resetJointState(self.robot_id, i, angle)

    raw_val = (
        gripper_state[0]
        if isinstance(gripper_state, (list, np.ndarray))
        else gripper_state
    )
    raw_val = np.clip(raw_val, 0.0, 1.0)
    angle = (raw_val * GRIPPER_ANGLE_SCALE) - GRIPPER_WIDTH_OFFSET
    for i, sign in zip(self.gripper_joints, self.gripper_signs):
      pybullet.resetJointState(self.robot_id, i, angle * sign)

  def _get_projection_matrix(self, K, width, height):
    """Builds a column-major OpenGL projection matrix from intrinsics K."""
    near, far = NEAR_PLANE, FAR_PLANE
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    columns = [
        [2.0 * fx / width, 0.0, 0.0, 0.0],
        [0.0, 2.0 * fy / height, 0.0, 0.0],
        [
            1.0 - 2.0 * cx / width,
            2.0 * cy / height - 1.0,
            (far + near) / (near - far),
            -1.0,
        ],
        [0.0, 0.0, 2.0 * far * near / (near - far), 0.0],
    ]
    matrix = []
    for column in columns:
      matrix.extend(column)
    return matrix

  def _render_raw(self, T_cam2world, K, width, height):
    """Renders raw OpenGL depth and link-segmentation buffers."""
    cam_pos = T_cam2world[:3, 3]
    view_matrix = pybullet.computeViewMatrix(
        cam_pos.tolist(),
        (cam_pos + T_cam2world[:3, 2]).tolist(),
        (-T_cam2world[:3, 1]).tolist(),
    )
    proj_matrix = self._get_projection_matrix(K, width, height)
    _, _, _, depth_buf, seg_buf = pybullet.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=self.render_mode,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    return depth_buf, seg_buf

  def render_depth(self, T_cam2world, K, width, height):
    depth_buf, _ = self._render_raw(T_cam2world, K, width, height)
    return metric_depth(depth_buf, height, width)

  def render_segmentation(self, T_cam2world, K, width, height):
    depth_buf, seg_buf = self._render_raw(T_cam2world, K, width, height)
    link_ids = (np.reshape(seg_buf, (height, width)).astype(np.int32) >> 24) - 1
    return link_ids, metric_depth(depth_buf, height, width)


_worker = None


def _start_worker(urdf, gpu):
  global _worker
  _worker = PyBulletRenderer(urdf, gpu)


def _render_frame(task):
  joint_positions, gripper_position, T_cam2world, K, width, height = task
  _worker.update_robot_pose(joint_positions, gripper_position)
  return _worker.render_depth(T_cam2world, K, width, height)


class RenderPool:
  """Multiprocessing pool for rendering URDF depth sequences across frames."""

  def __init__(self, renderer, workers):
    self.pool = multiprocessing.get_context("spawn").Pool(
        workers, _start_worker, (renderer.urdf, renderer.gpu)
    )

  def render(self, robot, T_cam2world, K, width, height):
    """Renders metric robot depth maps across all frames for one camera."""
    tasks = [
        (
            robot["joint_positions"][t],
            robot["gripper_positions"][t],
            T_cam2world[t],
            K,
            width,
            height,
        )
        for t in range(len(T_cam2world))
    ]
    return np.stack(self.pool.map(_render_frame, tasks, chunksize=8))

  def render_cameras(self, episode, poses):
    return {
        cam_id: self.render(
            episode["robot"],
            poses[cam_id]["T_cam2world"],
            cam_data["K"],
            *cam_data["raw_depth"].shape[:0:-1]
        )
        for cam_id, cam_data in episode["camera"].items()
    }

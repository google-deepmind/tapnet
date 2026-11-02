#!/bin/bash
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

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v ffmpeg >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y ffmpeg
fi

pip install -r requirements.txt

if ! python -c \
    "import pybullet, sys; sys.exit(0 if pybullet.isNumpyEnabled() else 1)" \
    2>/dev/null; then
  pip install --force-reinstall --no-deps --no-binary pybullet \
      --no-build-isolation --no-cache-dir pybullet
fi

CUDNN_LIB="$(python -c 'import os, site; print(os.path.join(site.getsitepackages()[0], "nvidia", "cudnn", "lib"))')"
if [ -n "${VIRTUAL_ENV:-}" ] && \
    ! grep -q "nvidia/cudnn/lib" "$VIRTUAL_ENV/bin/activate"; then
  echo "export LD_LIBRARY_PATH=\"$CUDNN_LIB:\$LD_LIBRARY_PATH\"" \
      >> "$VIRTUAL_ENV/bin/activate"
fi

POINTWORLD_COMMIT="05484826dfef74cbe278a3974179a5a16705d35d"
URDF_OUT="assets/franka_description/franka_panda_robotiq_2f85_og.urdf"
if [ ! -f "$URDF_OUT" ]; then
  (
    tmp_pw="$(mktemp -d)"
    trap 'rm -rf "$tmp_pw"' EXIT
    git init -q "$tmp_pw"
    git -C "$tmp_pw" remote add origin https://github.com/NVlabs/PointWorld.git
    git -C "$tmp_pw" sparse-checkout init --cone
    git -C "$tmp_pw" sparse-checkout set assets/franka_description
    git -C "$tmp_pw" fetch -q --depth 1 --filter=blob:none origin \
        "$POINTWORLD_COMMIT"
    git -C "$tmp_pw" checkout -q FETCH_HEAD
    mkdir -p assets/franka_description
    cp -r "$tmp_pw/assets/franka_description/meshes" assets/franka_description/
    python - "$tmp_pw/assets/franka_description/franka_panda_robotiq_2f85.urdf" \
        "$URDF_OUT" <<'EOF'
import sys

with open(sys.argv[1], "r", newline="") as f:
  text = f.read().replace("\r\n", "\n")

replacements = [
    (
        '  <!-- <link name="camera_mount_link">\n'
        '    <visual>\n'
        '      <origin xyz="0.0 -0.28 -0.009"',
        '  <link name="camera_mount_link">\n'
        '    <visual>\n'
        '      <origin xyz="0.0 -0.28 -0.009"',
    ),
    (
        '  </link> -->\n\n'
        '  <!-- Attach camera mount to panda_link8'
        ' (larger mount to filter out close-by points) -->\n'
        '  <link name="camera_mount_link">',
        '  </link>\n\n'
        '  <!-- Attach camera mount to panda_link8'
        ' (larger mount to filter out close-by points) -->\n'
        '  <!-- <link name="camera_mount_link">',
    ),
    (
        '        <mesh scale="0.0013 0.0013 0.0013"'
        ' filename="meshes/droid_hand_camera_part_2.stl"/>\n'
        '      </geometry>\n'
        '    </collision>\n'
        '  </link>',
        '        <mesh scale="0.0013 0.0013 0.0013"'
        ' filename="meshes/droid_hand_camera_part_2.stl"/>\n'
        '      </geometry>\n'
        '    </collision>\n'
        '  </link> -->',
    ),
]
for src, dst in replacements:
  if src not in text:
    raise RuntimeError(f"Expected URDF snippet not found: {src[:40]}")
  text = text.replace(src, dst, 1)

with open(sys.argv[2], "w", newline="\n") as f:
  f.write(text)
EOF
  )
fi

if [ "${1:-}" = "--no-depth" ]; then
  exit 0
fi

S2M2_COMMIT="8f3bd785df593619d955203c6f2c741bef6e31df"
if [ ! -d third_party/s2m2/.git ]; then
  git clone https://github.com/junhong-3dv/s2m2.git third_party/s2m2
  git -C third_party/s2m2 checkout "$S2M2_COMMIT"
fi

pip install -r requirements-depth.txt

if ! python -c "import pyzed.sl" 2>/dev/null; then
  (
    tmp_zed="$(mktemp -d)"
    trap 'rm -rf "$tmp_zed"' EXIT
    cd "$tmp_zed"
    if [ ! -f /usr/local/zed/get_python_api.py ]; then
      sudo apt-get update -qq && sudo apt-get install -y zstd
      wget -O zed_sdk.run \
          "https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22"
      chmod +x zed_sdk.run
      ./zed_sdk.run silent runtime_only skip_tools
    fi
    python /usr/local/zed/get_python_api.py
  )
fi

mkdir -p third_party/s2m2/weights third_party/segment_anything/weights
wget -nc -P third_party/s2m2/weights \
    "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -P third_party/segment_anything/weights \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

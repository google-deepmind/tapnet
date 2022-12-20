#!/bin/bash
# Copyright 2022 DeepMind Technologies Limited
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


virtualenv -p python3 /tmp/venv
source /tmp/venv/bin/activate

pip3 install -r requirements.txt

python visualize.py --input_path=tapvid_davis.pkl --output_path=tapvid_davis.mp4

python visualize.py --input_path=tapvid_rgb_stacking.pkl --output_path=tapvid_rgb_stacking.mp4

python generate_tapvid.py --input_csv_path=tapvid_kinetics.csv --output_base_path=tapvid_kinetics --video_root_path=kinetics_videos
python visualize.py --input_path=tapvid_kinetics/0001_of_0010.pkl --output_path=tapvid_kinetics.mp4

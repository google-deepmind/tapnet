#!/bin/bash
<<<<<<< HEAD
=======
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

>>>>>>> cb3ef0ce00547001f108aa7094a4d0f2071e4079

virtualenv -p python3 /tmp/venv
source /tmp/venv/bin/activate

pip3 install -r requirements.txt

python visualize_pickle.py --input_pkl_path=tapvid_davis.pkl --output_path=tapvid_davis.mp4

python visualize_kinetics.py --input_csv_path=tapvid_kinetics.csv --input_video_dir=kinetics_videos --output_path=tapvid_kinetics.mp4

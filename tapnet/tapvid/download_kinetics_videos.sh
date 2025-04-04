#!/bin/bash
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


git clone https://github.com/cvdfoundation/kinetics-dataset.git
mkdir kinetics_videos
cd kinetics_videos
wget https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt
bash ../kinetics-dataset/download.sh k700_2020_val_path.txt
bash ../kinetics-dataset/extract.sh k700_2020_val_path.txt
rm -f k700_val_*
rm -f k700_2020_val_path.txt
cd ..
rm -rf kinetics-dataset

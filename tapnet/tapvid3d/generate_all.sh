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


set -x

debug=0

# Parse debug option
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -d|--debug)
            debug=1
            ;;
        *)
            echo "Unknown option: $1"
            ;;
    esac
    shift
done

if [[ $debug -eq 1 ]]; then
    PYTHON_DEBUG="True"
else
    PYTHON_DEBUG="False"
fi

python3 -m venv tapvid3d
source tapvid3d/bin/activate

# Download the ADT data and annotations
ADT_OUTPUT_DIRECTORY="tapvid3d_dataset/adt/"
mkdir -p $ADT_OUTPUT_DIRECTORY
python3 -m tapnet.tapvid3d.annotation_generation.generate_adt --output_dir=$ADT_OUTPUT_DIRECTORY --debug=$PYTHON_DEBUG --split=all

# Download the Panoptic Studio data and annotations
PSTUDIO_OUTPUT_DIRECTORY="tapvid3d_dataset/pstudio/"
mkdir -p $PSTUDIO_OUTPUT_DIRECTORY
python3 -m tapnet.tapvid3d.annotation_generation.generate_pstudio --output_dir=$PSTUDIO_OUTPUT_DIRECTORY --debug=$PYTHON_DEBUG --split=all

# Download the Waymo Open / DriveTrack data and annotations
DT_OUTPUT_DIRECTORY="tapvid3d_dataset/drivetrack/"
mkdir -p $DT_OUTPUT_DIRECTORY
python3 -m tapnet.tapvid3d.annotation_generation.generate_drivetrack --output_dir=$DT_OUTPUT_DIRECTORY --debug=$PYTHON_DEBUG --split=all

deactivate

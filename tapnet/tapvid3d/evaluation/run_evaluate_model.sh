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


# Expects:
# (1) to be run from this directory! `./run_evaluate_model.sh``
# (2) that a copy of the generated dataset is in TAPVID3D_DIR, defined below
#     and should be modified as necessary, depending on where you stored the
#     generated dataset. Format is as described in the README.md, in summary:
#     a folder with drivetrack/, adt/, pstudio/ subfolders, each containing
#     *.npz files, one per video containing ground truth 3D tracks and other
#     metadata.

TAPVID3D_DIR="/tmp/datasets/tapvid3d/"

python3 -m venv eval_tapvid3d
source eval_tapvid3d/bin/activate
pip install absl-py==2.1.0 tqdm==4.66.4 numpy==2.0.0 pillow=10.4.0

cd ../..
pip install .
cd tapvid3d

# First, download the SpaTracker predictions on the DriveTrack subset from GCP
PREDICTIONS_DIR="/tmp/model_outputs/spatracker/"
mkdir -p "$PREDICTIONS_DIR"

# Get the DriveTrack filenames
PYTHON_SCRIPT=$(cat <<END
from tapnet.tapvid3d.splits import tapvid3d_splits

def get_filenames():
    filenames = tapvid3d_splits.get_full_eval_files('drivetrack')
    for filename in filenames:
        print(filename)

get_filenames()
END
)

GCP_PREDS_DIR="https://storage.googleapis.com/dm-tapnet/tapvid3d/release_predictions_files/spatracker/drivetrack/"

# Run the Python script and get the list of filenames
FILENAMES=$(python3 -c "$PYTHON_SCRIPT")

# Loop through the filenames, prepend the URL prefix, and download each file
for FILENAME in $FILENAMES; do
    URL="${GCP_PREDS_DIR}${FILENAME}"
    OUTPUT_PATH="${PREDICTIONS_DIR}${FILENAME}"
    echo "Downloading $URL to $OUTPUT_PATH..."
    curl -o "$OUTPUT_PATH" "$URL"
done

python3 evaluate_model.py \
  --tapvid3d_dir=$TAPVID3D_DIR \
  --tapvid3d_predictions=$PREDICTIONS_DIR \
  --use_minival=True \
  --data_sources_to_evaluate=drivetrack \
  --debug=False

deactivate

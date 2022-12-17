#!/bin/bash

virtualenv -p python3 /tmp/venv
source /tmp/venv/bin/activate

pip3 install -r requirements.txt

python visualize_pickle.py --input_pkl_path=tapvid_davis.pkl --output_path=tapvid_davis.mp4

python visualize_kinetics.py --input_csv_path=tapvid_kinetics.csv --input_video_dir=kinetics_videos --output_path=tapvid_kinetics.mp4

#!/bin/bash

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

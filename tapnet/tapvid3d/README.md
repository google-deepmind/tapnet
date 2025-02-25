<!-- mdlint off(WHITESPACE_LINE_LENGTH) -->
# Tracking Any Point in 3D (TAPVid-3D)

[Skanda Koppula](https://skoppula.com/), [Ignacio Rocco](https://www.irocco.info/), [Yi Yang](https://yangyi02.github.io/), [Joe Heyward](https://uk.linkedin.com/in/joe-heyward-71623595), [João Carreira](https://uk.linkedin.com/in/jo%C3%A3o-carreira-56238a7), [Andrew Zisserman](https://www.robots.ox.ac.uk/~az/), [Gabriel Brostow](http://www0.cs.ucl.ac.uk/staff/G.Brostow/), [Carl Doersch](http://www.carldoersch.com/)

**[Google DeepMind](https://deepmind.google/)**, **[University College London](http://vis.cs.ucl.ac.uk/home/), [University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**

### [`TAPVid-3D Website`](https://tapvid3d.github.io/) [`TAPVid-3D Paper`](https://arxiv.org/abs/2407.05921) [`Colab to Visualize Samples`](https://colab.research.google.com/github/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/colab/load_and_visualize_tapvid3d_samples.ipynb)

TAPVid-3D is a dataset and benchmark for evaluating the task of long-range
Tracking Any Point in 3D (TAP-3D).

The benchmark features 4,000+ real-world videos, along with their metric 3D
position point trajectories and camera extrinsics. The dataset is contains three
different video sources, and spans a variety of object types, motion patterns,
and indoor and outdoor environments. This repository folder contains the code to
 download and generate these annotations and dataset samples to view.

**Note that in order to use the dataset, you must accept the licenses of the
constituent original video data sources that you use!** In particular, you must
adhere to the terms of service outlined in:

1. [Aria Digital Twin](https://www.projectaria.com/datasets/adt/license/)
2. [Waymo Open Dataset](https://waymo.com/open/terms/)
3. [Panoptic Studio](http://domedb.perception.cs.cmu.edu/)

To measure performance on the TAP-3D task, we formulated metrics that extend the
Jaccard-based metric used in 2D TAP to handle the complexities of ambiguous
depth scales across models, occlusions, and multi-track spatio-temporal
smoothness. Our implementation of these metrics (3D-AJ, APD, OA) can be found in
`tapvid3d_metrics.py`.

For more details, including the performance of multiple baseline 3D point
tracking models on the benchmark, please see the paper:
[TAPVid-3D:A Benchmark for Tracking Any Point in 3D](http://arxiv.org).

### Getting Started: Installing

(If you want to generate the dataset and want to use CUDA for running semantic segmentation, first install a CUDA-enabled PyTorch with `pip3 install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118`)

For generating the dataset, install the Tracking Any Point repo with:

`pip install "git+https://github.com/google-deepmind/tapnet.git"[tapvid3d_eval,tapvid3d_generation]`

If you only want to use the metrics, install with:

`pip install "git+https://github.com/google-deepmind/tapnet.git"[tapvid3d_eval]`

For a local editable installation, clone the repo and use:

`pip install -e .[tapvid3d_eval,tapvid3d_generation]`

or

`pip install -e .[tapvid3d_eval]`.

### How to Download and Generate the Dataset

Scripts to download and generate the annotations of each of the datasets can be
found in the `annotation_generation` subdirectory, and can be run as:

```
python3 -m tapnet.tapvid3d.annotation_generation.generate_adt --help
python3 -m tapnet.tapvid3d.annotation_generation.generate_pstudio --help
python3 -m tapnet.tapvid3d.annotation_generation.generate_drivetrack --help
```

Because of license restrictions in distribution of the underlying source videos
for Aria Digital Twin, you will need to accept their licence terms and download
the ADT dataset by first getting the cdn json file from
[Project Aria Explorer](https://explorer.projectaria.com/?v=%22Aria+Digital+Twin%22),
and downloading the ADT `main_vrs`, `main_groundtruth`, `segmentation` and `depth` files with:

`aria_dataset_downloader --cdn_file /PATH_TO/Aria_Digital_Twin_1720774989.json -o /OUTPUT_PATH -d 0 6 7 8`

To run all generation scripts, follow the instructions and run the commands in
`generate_all.sh`. This will generate all the `*.npz` files, and place them into
a new folder, `tapvid3d_dataset`. To generate only the `minival` split,
replace `--split=all` in this script with `--split=minival`.

To test things are working before full generation, you can run `./run_all.sh
--debug`. This runs generates exactly one `*.npz`/video annotation per data
source.

### Data format

Once the benchmark files are fully generated, you will have roughly 4,500
`*.npz` files, each one with exactly one dataset video+annotation. Each `*.npz`
file, contains:

*  `images_jpeg_bytes`: tensor of shape [`# of frames`, `height`, `width`, 3],
  each frame stored as JPEG bytes that must be decoded

*  `intrinsics`: (fx, fy, cx, cy) camera intrinsics of the video

*  `tracks_xyz`: tensor of shape (`# of frames`, `# of point tracks`, 3),
  representing the 3D point trajectories and the last dimension is the `(x, y,
  z)` point position in meters.

*  `visibility`: tensor of shape (`# of frames`, `# of point tracks`),
  representing the visibility of each point along its trajectory

*  `queries_xyt`: tensor of shape (`# of point tracks`, 3), representing the
  query point used in the benchmark as the initial given point to track. The
  last dimension is given in `(x, y, t)`, where `x,y` is the pixel location of
  the query point and `t` is the query frame.

*  `extrinsics_w2c`: for videos with a moving camera (videos from the Waymo
  Open Dataset and ADT), we provide camera extrinsics in the form of a world
  to camera transform, a 4x4 matrix consisting of the rotation matrix and
  translation matrix. This field is NOT present in the `pstudio` *.npz files,
  because in Panoptic Studio, the camera is static.

### Getting Started: Evaluating Your Own 3D Tracking Model

To get started using this dataset to benchmark your own 3D tracking model, check
out `evaluate_model.py`. This script reads the ground truth track prediction,
and predicted tracks and computes the TAPVid-3D evaluation metrics for those
predictions. An example of running this is provided in `run_evaluate_model.sh`,
which computes metrics for our precomputed outputs from
[SpatialTracker](https://henry123-boy.github.io/SpaTracker/) on the `minival`
split, for only the `DriveTrack` subset. `evaluate_model.py`
takes as input a folder with subdirectories (corresponding to up to 3 data
sources [among PStudio/DriveTrack/ADT]). Each
subdirectory should contain a list of `*.npz` files
(one per TAPVid-3D video, each containing
`tracks_XYZ` and `visibility` in the format above).
Running this will return all TAP-3D metrics
described in the paper (these are implemented in `tapvid3d_metrics.py`).

### Visualizing Samples in Colab

You can view samples of the dataset, using a public Colab demo:
<a target="https://colab.research.google.com/github/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/colab/load_and_visualize_tapvid3d_samples.ipynb" href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TAPVid-3D Colab Visualization"/></a>.

## Citing this Work

If you find this work useful, consider citing the manuscript:

```
@inproceedings{koppula2024tapvid3d,
   title={TAPVid-3D: A Benchmark for Tracking Any Point in 3D},
   author={Skanda Koppula and Ignacio Rocco and Yi Yang and Joe Heyward and João Carreira and Andrew Zisserman and Gabriel Brostow and Carl Doersch},
   booktitle={NeurIPS},
   year={2024},
}
```

## License and Disclaimer

Copyright 2024 Google LLC

All software here is licensed under the Apache License, Version 2.0 (Apache
2.0); you may not use this file except in compliance with the Apache 2.0
license. You may obtain a copy of the Apache 2.0 license at:

https://www.apache.org/licenses/LICENSE-2.0

All other materials, excepting the source videos and artifacts used to generate
annotations, are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

The Aria Digital Twin, Waymo Open Dataset, and Panoptic Studio datasets all
have individual usage terms and licenses that must be adhered to when using any
software or materials from here.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

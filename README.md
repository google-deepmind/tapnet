# TAP-Vid: A Benchmark for Tracking Any Point in a Video
## Introduction

TAP-Vid is a dataset of videos along with point tracks, either manually annotated or obtained from a simulator. The aim is to evaluate tracking of any trackable point on any solid physical surface. Algorithms receive a single query point on some frame, and must produce the rest of the track, i.e., including where that point has moved to (if visible), and whether it is visible, on every other frame. This requires point-level precision (unlike prior work on box and segment tracking) potentially on deformable surfaces (unlike structure from motion) over the long term (unlike optical flow) on potentially any object (i.e. class-agnostic, unlike prior class-specific keypoint tracking on humans). Here's an example of what's annotated on one video of the DAVIS dataset:

/*:
  To include this video, one should go to github and drag it into the edit page (verified that works)

  https://storage.googleapis.com/dm-tapnet/content/davis_ground_truth/soapbox.mp4
 */

We also include a point tracking
model TAP-Net with code to train it using a combination
of sim2real and self-supervised learning. TAP-Net outperforms
both optical flow and structure-from-motion methods on the
TAP-Vid benchmark while achieving state-of-the-art performance
on unsupervised human keypoint tracking on JHMDB, even though
the model tracks points on clothes and skin rather than the
joints as intended by the benchmark.

## Downloading and Using the Dataset
For DAVIS and RGB-Stacking, the videos are contained in a simple pickle file; for DAVIS, this contains a simple dict, where each key is a DAVIS video title, and the contents are the video (4D uint8 tensor), the points (float32 tensor with 3 axes; the first is point id, the second is time, and the third is x/y), and the occlusions (bool tensor with 2 axies; the first is point id, the second is time). RGB-Stacking is the same, except there's no video titles, so it's a simple list of these structures rather than a dict. The downloads are given above.
For Kinetics, we cannot distribute the raw videos, so instructions for assembling the above data structures are given below.
### Downloading the Kinetics videos
We expect the raw clips from Kinetics700-2020 validation set to be downloaded and stored in a local folder ```<video_root_path>```. The clips should be stored as MP4, following the name pattern
 ```f'{youtube_id}_{start_time_sec:06}_{end_time_sec:06}', e.g. 'abcdefghijk_000010_000020.mp4'```.
Clips can be stored in any subfolder within the ```<video_root_path>```. The most common pattern is to store it as ```<video_root_path>/<label_name>/<clip_name>```.
### Processing the clips
Once the validation clips have been downloaded, a pickle file containing all the information can be generated using the provided script:
```
bash
python3 -m pip install -r requirements.txt
python3 generate_tapvid.py \
  --csv_path=<path_to_tapvid_kinetics.csv> \
  --output_base_path=<path_to_output_pickle_folder> \
  --video_root_path=<path_to_raw_videos_root_folder> \
  --alsologtostderr
```
### Visualizing annotations
We also provide a script generating an MP4 with the points painted on top of the frames. The script will work with any of the pickle files (Kinetics Tapnet, Davis or Robotics). A random clip is chosen from all the available ones and all the point tracks are painted.
```
bash
python3 -m pip install -r requirements.txt
python3 visualize.py \
  --input_path=<path_to_the_pickle_file.pkl> \
  --output_path=<path_to_the_output_video.mp4> \
  --alsologtostderr
```

### Visualized Ground Truth Examples
To demonstrate the datasets we have created, we include the full TAP-Vid-DAVIS (30 videos) (old version), as well as 10 examples each from the synthetic TAP-Vid-Kubric and TAP-Vid-RGB-Stacking datasets (old version).

### Comparison of Tracking With and Without Optical Flow
When annotating videos, we interpolate between the sparse points that the annotators choose by finding tracks which minimize the discrepancy with the optical flow while still connecting the chosen points. To validate that this is indeed improving results, we annotated several DAVIS videos twice and compare them side by side, once using the flow-based interpolation, and again using a naive linear interpolation, which simply moves the point at a constant velocity between points.

### Hosting and Maintenance
The majority of the code as well as standalone annotations will be hosted in DeepMind’s Github once we are prepared for a public announcement, which is currently scheduled to happen late in July. For our datasets which include videos (DAVIS and robotics), the default files are too large for Github, and so these will be hosted in a Google cloud bucket. DeepMind plans to ensure the availability of these repositories in the long term, and we expect maintenance to be minimal as simple python readers are sufficient to use the data.


## TAP-Net Installation

Install ffmpeg on your machine:

```sudo apt update```

```sudo apt install ffmpeg```

Install OpenEXR:

```sudo apt-get install libopenexr-dev```

Clone the repository:

```git clone https://github.com/deepmind/tapnet.git```

Add current path (parent directory of where TapNet is installed)
to ```PYTHONPATH```:

```export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH```

Switch to the project directory:

```cd tapnet```

Install kubric as a subdirectory:

```git clone https://github.com/google-research/kubric.git```

Install requirements:

```pip install -r requirements.txt```

If you want to use CUDA, make sure you install the drivers and a version
of JAX that's compatible with your CUDA and CUDNN versions.
Refer to
[the jax manual](https://github.com/google/jax#pip-installation-gpu-cuda)
to install JAX version with CUDA.

## Usage

The configuration file is located at: ```./configs/tapnet_config.py```.

You can modify it for your need or create your own config file following
the example of ```tapnet_config.py```.

To launch experiment run the command:

```python ./experiment.py --config ./configs/tapnet_config.py```

## Evaluation

You can run evaluation for a particular dataset using the command:

```python3 ./tapnet/experiment.py --config ./tapnet/configs/tapnet_config.py --jaxline_mode=eval_davis --config.checkpoint_dir=/path/to/checkpoint/dir/```

Available eval datasets are listed in `supervised_point_prediction.py`.

`/path/to/checkpoint/dir/` must contain a file checkpoint.npy that's loadable
using our NumpyFileCheckpointer.

## A note on coordinates

In our storage datasets, (x, y) coordinates are typically in normalized raster
coordinates: i.e., (0, 0) is the upper-left corner of the upper-left pixel, and
(1, 1) is the lower-left corner of the lower-right pixel.  Our code, however,
immediately converts these to regular raster coordinates, matching the output of
the Kubric reader: (0, 0) is the upper-left corner of the upper-left pixel,
while (h, w) is the lower-right corner of the lower-right pixel, where h is the
image height in pixels, and w is the respctive width.

When working with 2D coordinates, we typically store them in the order (x, y).
However, we typically work with 3D coordinates in the order (t, y, x), where
y and x are raster coordinates as above, but t is in frame coordinates, i.e.
0 refers to the first frame, and 0.5 refers to halfway between the first and
second frames.  Please take care with this: one pixel error can make a
difference according to our metrics.



## Citing this work

Please use the following bibtex entry to cite ```TapNet-Vid```:

```
@inproceedings{doersch2022tapvid,
  author = {Doersch, Carl and Gupta, Ankush and Markeeva, Larisa and
            Continente, Adria Recasens and Smaira, Kucas and Aytar, Yusuf and
            Carreira, Joao and Zisserman, Andrew and Yang, Yi},
  title = {TAP-Vid: A Benchmark for Tracking Any Point in a Video},
  booktitle={NeurIPS Datasets Track},
  year = {2022},
}
```


## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode
In particular the annotations of TAP-Vid, as well as the RGB-Stacking videos, are released under a [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/).
The original source videos of DAVIS come from the val set, and are also licensed under creative commons licenses per their creators; see the [DAVIS dataset](https://davischallenge.org/davis2017/code.html) for details. Kinetics videos are publicly available on YouTube, but subject to their own individual licenses. See the [Kinetics dataset webpage](https://www.deepmind.com/open-source/kinetics) for details.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

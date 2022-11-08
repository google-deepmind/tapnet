# TAP-Vid: A Benchmark for Tracking Any Point in a Video

Full paper available at [https://arxiv.org/abs/2211.03726](https://arxiv.org/abs/2211.03726)

## Introduction

TAP-Vid is a dataset of videos along with point tracks, either manually annotated or obtained from a simulator. The aim is to evaluate tracking of any trackable point on any solid physical surface. Algorithms receive a single query point on some frame, and must produce the rest of the track, i.e., including where that point has moved to (if visible), and whether it is visible, on every other frame. This requires point-level precision (unlike prior work on box and segment tracking) potentially on deformable surfaces (unlike structure from motion) over the long term (unlike optical flow) on potentially any object (i.e. class-agnostic, unlike prior class-specific keypoint tracking on humans). Here's an example of what's annotated on one video of the DAVIS dataset:


https://user-images.githubusercontent.com/15641194/199806865-e881ffe9-24fc-4fa6-98ed-5c85d363f49e.mp4



For our full benchmark incorporates 4 datasets: the 30 videos of the DAVIS-val set, more than 30,000 points on 1000 videos from the Kinetics dataset, 50 synthetic robotics videos with perfect ground truth, and point annotations on the large-scale synthetic Kubric dataset for training (see [here](https://github.com/google-research/kubric/tree/main/challenges/point_tracking).  For more examples, we have the full [TAP-Vid-DAVIS](https://storage.googleapis.com/dm-tapnet/content/davis_ground_truth_v2.html) as well as 10 examples each from the synthetic [TAP-Vid-Kubric](https://storage.googleapis.com/dm-tapnet/content/kubric_ground_truth.html) and [TAP-Vid-RGB-Stacking](https://storage.googleapis.com/dm-tapnet/content/rgb_stacking_ground_truth_v2.html) datasets.


We also include a point tracking
model TAP-Net, with code to train it on Kubric data.
TAP-Net outperforms
both optical flow and structure-from-motion methods on the
TAP-Vid benchmark while achieving state-of-the-art performance
on unsupervised human keypoint tracking on JHMDB, even though
the model tracks points on clothes and skin rather than the
joints as intended by the benchmark.

## Downloading and Using the Dataset

There are three dataset files to download: [DAVIS](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip), [Kinetics](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip), and [RGB-stacking](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip).
For DAVIS and RGB-Stacking, the videos are contained in a simple pickle file; for DAVIS, this contains a simple dict, where each key is a DAVIS video title, and the contents are the video (4D uint8 tensor), the points (float32 tensor with 3 axes; the first is point id, the second is time, and the third is x/y), and the occlusions (bool tensor with 2 axies; the first is point id, the second is time). RGB-Stacking is the same, except there's no video titles, so it's a simple list of these structures rather than a dict. The downloads are given above.
For Kinetics, we cannot distribute the raw videos, so instructions for
assembling the above data structures are given [here](data/README.md).

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

### Evaluating on TAP-Vid

[`evaluation_datasets.py`](evaluation_datasets.py) is intended to be a
stand-alone, copy-and-pasteable reader and evaluator, which depends only
on numpy and other basic tools.  Tensorflow is required only for reading Kubric
(which provides a tensorflow reader by default) as well as file operations,
which should be straightforward to replace for systems without Tensorflow.

For each dataset, there is a basic reader which will produce examples, dicts of
numpy arrays containing the video, the query points, the target points, and the
occlusion flag.  Evaluation datasets may be used with one of two possible values
for `query_mode`: `strided` (each trajectory is queried multiple times, with
a fixed-length stride between queries)  or `first` (each trajectory is queried
once, with only the first visible point on the query).  For details on outputs,
see the documentation for `sample_queries_strided` and `sample_queries_first`.

To compute metrics, use `compute_tapvid_metrics` in the same file.  This
computes results on each batch; the final metrics for the paper can be computed
by simple averaging across all videos in the dataset.  See the documentation for
more details.

Note that the outputs for a single query point *should not depend on the other
queries defined in the batch*: that is, the outputs should be the same whether
the queries are passed one at a time or all at once.  This is important because
the other queries may leak information about how pixels are grouped and how they
move.  This property is not enforced in the current evaluation code, but
algorithms which violate this principle should not be considered valid
competitors on this benchmark.

Our readers also supply videos resized at 256-by-256.  If algorithms can handle
it, we encourage using full-resolution videos instead; we anticipate that
predictions on such videos would be scaled to match a 256-by-256 resolution
before computing metrics.
Such predictions would, however, be evaluated as a separate category: we don't
consider them comparable to those produced from lower-resolution videos.

### A note on coordinates

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

## Comparison of Tracking With and Without Optical Flow
When annotating videos, we interpolate between the sparse points that the annotators choose by finding tracks which minimize the discrepancy with the optical flow while still connecting the chosen points. To validate that this is indeed improving results, we annotated several DAVIS videos twice and [compare them side by side](https://storage.googleapis.com/dm-tapnet/content/flow_tracker.html), once using the flow-based interpolation, and again using a naive linear interpolation, which simply moves the point at a constant velocity between points.

## Installation for TAP-Net training and inference

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
using our NumpyFileCheckpointer.  You can download a checkpoint
[here](https://storage.googleapis.com/dm-tapnet/checkpoint.npy), which
was obtained via the open-source version of the code, and should closely match
the one used to write the paper.


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

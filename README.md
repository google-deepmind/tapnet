# Tracking Any Point (TAP)

[[`TAP-Vid`](https://tapvid.github.io/)] [[`TAPIR`](https://deepmind-tapir.github.io/)] [[`RoboTAP`](https://robotap.github.io/)] [[`Blog Post`](https://deepmind-tapir.github.io/blogpost.html)] [[`BootsTAP`](https://bootstap.github.io/)]

https://github.com/google-deepmind/tapnet/assets/4534987/9f66b81a-7efb-48e7-a59c-f5781c35bebc

Welcome to the official Google Deepmind repository for Tracking Any Point (TAP), home of the TAP-Vid Dataset, our top-performing TAPIR model, and our RoboTAP extension.

- [TAP-Vid](https://tapvid.github.io) is a benchmark for models that perform this task, with a collection of ground-truth points for both real and synthetic videos.
- [TAPIR](https://deepmind-tapir.github.io) is a two-stage algorithm which employs two stages: 1) a matching stage, which independently locates a suitable candidate point match for the query point on every other frame, and (2) a refinement stage, which updates both the trajectory and query features based on local correlations. The resulting model is fast and surpasses all prior methods by a significant margin on the TAP-Vid benchmark.
- [RoboTAP](https://robotap.github.io) is a system which utilizes TAPIR point tracks to execute robotics manipulation tasks through efficient imitation in the real world. It also includes a dataset with ground-truth points annotated on real robotics manipulation videos.
- [BootsTAP](https://bootstap.github.io) (or Bootstrapped Training for TAP) uses a large dataset of unlabeled, real-world video to improve tracking accuracy. Specifically, the model is trained to give consistent predictions across different spatial transformations and corruptions of the video, as well as different choices of the query points. We apply it to TAPIR to create BootsTAPIR, which is architecturally similar to TAPIR but substantially outperforms it on TAP-Vid.

This repository contains the following:

- [TAPIR Demos](#tapir-demos) for both online **colab demo** and offline **real-time demo** by cloning this repo
- [TAP-Vid Benchmark](#tap-vid-benchmark) for both evaluation **dataset** and evaluation **metrics**
- [RoboTAP](#roboTAP-benchmark-and-point-track-based-clustering) for both evaluation **dataset** and point track based clustering code
- [BootsTAP](#colab-demo) for further improved BootsTAPIR model using large scale **semi-supervised bootstrapped** learning
- [Checkpoints](#download-checkpoints) for both TAP-Net (the baseline presented in the TAP-Vid paper), TAPIR and BootsTAPIR **pre-trained** model weights in both **Jax** and **PyTorch**
- [Instructions](#tap-net-and-tapir-training-and-inference) for both **training** TAP-Net (the baseline presented in the TAP-Vid paper) and TAPIR on Kubric

## TAPIR Demos

The simplest way to run TAPIR is to use our colab demos online.  You can also
clone this repo and run TAPIR on your own hardware, including a real-time demo.

### Colab Demo

You can run colab demos to see how TAPIR works. You can also upload your own video and try point tracking with TAPIR.
We provide a few colab demos:

1. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Offline TAPIR"/></a> **Standard TAPIR**: This is the most powerful TAPIR / BootsTAPIR model that runs on a whole video at once. We mainly report the results of this model in the paper.
2. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/causal_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Online TAPIR"/></a> **Online TAPIR**: This is the sequential causal TAPIR / BootsTAPIR model that allows for online tracking on points, which can be run in real-time on a GPU platform.
3. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_rainbow_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TAPIR Rainbow Visualization"/></a> **Rainbow Visualization**: This visualization is used in many of our teaser videos: it does automatic foreground/background segmentation and corrects the tracks for the camera motion, so you can visualize the paths objects take through real space.
4. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Offline PyTorch TAPIR"/></a> **Standard PyTorch TAPIR**: This is the TAPIR / BootsTAPIR model re-implemented in PyTorch, which contains the exact architecture & weights as the Jax model.
4. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_causal_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Online PyTorch TAPIR"/></a> **Online PyTorch TAPIR**: This is the sequential causal BootsTAPIR model re-implemented in PyTorch, which contains the exact architecture & weights as the Jax model.

### Live Demo

Clone the repository:

```git clone https://github.com/deepmind/tapnet.git```

Switch to the project directory:

```cd tapnet```

Install the `tapnet` python package (and its requirements for running inference):

```pip install .```

Download the checkpoint

```bash
mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```

Add current path (parent directory of where TapNet is installed)
to ```PYTHONPATH```:

```export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH```

If you want to use CUDA, make sure you install the drivers and a version
of JAX that's compatible with your CUDA and CUDNN versions.
Refer to
[the jax manual](https://github.com/google/jax#installation)
to install the correct JAX version with CUDA.

You can then run a pretrained causal TAPIR model on a live camera and select points to track:

```bash
cd ..
python3 ./tapnet/live_demo.py \
```

In our tests, we achieved ~17 fps on 480x480 images on a quadro RTX 4000.

## TAP-Vid Benchmark

https://github.com/google-deepmind/tapnet/assets/4534987/ff5fa5e3-ed37-4480-ad39-42a1e2744d8b

[TAP-Vid](https://tapvid.github.io) is a dataset of videos along with point tracks, either manually annotated or obtained from a simulator. The aim is to evaluate tracking of any trackable point on any solid physical surface. Algorithms receive a single query point on some frame, and must produce the rest of the track, i.e., including where that point has moved to (if visible), and whether it is visible, on every other frame. This requires point-level precision (unlike prior work on box and segment tracking) potentially on deformable surfaces (unlike structure from motion) over the long term (unlike optical flow) on potentially any object (i.e. class-agnostic, unlike prior class-specific keypoint tracking on humans).

Our full benchmark incorporates 4 datasets: 30 videos from the [DAVIS val set](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip), 1000 videos from the [Kinetics val set](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip), 50 synthetic [Deepmind Robotics videos](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip) for evaluation, and (almost infinite) point track ground truth on the large-scale synthetic [Kubric dataset](https://github.com/google-research/kubric/tree/main/challenges/point_tracking) for training.

For more details of downloading and visualization of the dataset, please see the [data section](https://github.com/deepmind/tapnet/tree/main/data).

We also include a point tracking model TAP-Net, with code to train it on Kubric dataset. TAP-Net outperforms both optical flow and structure-from-motion methods on the TAP-Vid benchmark while achieving state-of-the-art performance on unsupervised human keypoint tracking on JHMDB, even though the model tracks points on clothes and skin rather than the joints as intended by the benchmark.

### Evaluating on TAP-Vid

[`evaluation_datasets.py`](tapvid/evaluation_datasets.py) is intended to be a
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

Our readers also supply videos resized at 256x256 resolution.  If algorithms can handle it, we encourage using full-resolution videos instead; we anticipate that
predictions on such videos would be scaled to match a 256x256 resolution
before computing metrics. Such predictions would, however, be evaluated as a separate category: we don't consider them comparable to those produced from lower-resolution videos.

### A Note on Coordinates

In our storage datasets, (x, y) coordinates are typically in normalized raster
coordinates: i.e., (0, 0) is the upper-left corner of the upper-left pixel, and
(1, 1) is the lower-right corner of the lower-right pixel.  Our code, however,
immediately converts these to regular raster coordinates, matching the output of
the Kubric reader: (0, 0) is the upper-left corner of the upper-left pixel,
while (h, w) is the lower-right corner of the lower-right pixel, where h is the
image height in pixels, and w is the respective width.

When working with 2D coordinates, we typically store them in the order (x, y).
However, we typically work with 3D coordinates in the order (t, y, x), where
y and x are raster coordinates as above, but t is in frame coordinates, i.e.
0 refers to the first frame, and 0.5 refers to halfway between the first and
second frames.  Please take care with this: one pixel error can make a
difference according to our metrics.

### Comparison of Tracking With and Without Optical Flow

When annotating videos for the TAP-Vid benchmark, we use a track assist algorithm interpolates between the sparse points that the annotators click, since requiring annotators to click every frame is prohibitively expensive.  Specifically, we find tracks which minimize the discrepancy with the optical flow while still connecting the chosen points. Annotators will then check the interpolations and repeat the annotation until they observe no drift.

To validate that this is a better approach than a simple linear interpolation between clicked points, we annotated several DAVIS videos twice and [compare them side by side](https://storage.googleapis.com/dm-tapnet/content/flow_tracker.html), once using the flow-based interpolation, and again using a naive linear interpolation, which simply moves the point at a constant velocity between points.

<a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/optical_flow_track_assist.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Point Track Annotation"/></a> **Flow assist point annotation**: You can run this colab demo to see how point tracks are annotated with optical flow assistance.

## RoboTAP Benchmark and Point Track based Video Segmentation

[RoboTAP](https://robotap.github.io/) is a following work of TAP-Vid and TAPIR that demonstrates point tracking models are important for robotics.

The [RoboTAP dataset](https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip) follows the same annotation format as TAP-Vid, but is released as an addition to TAP-Vid. In terms of domain, RoboTAP dataset is mostly similar to TAP-Vid-RGB-Stacking, with a key difference that all robotics videos are real and manually annotated. Video sources and object categories are also more diversified. The benchmark dataset includes 265 videos, serving for evaluation purpose only.

For more details of downloading and visualization of the dataset, please see the [data section](https://github.com/deepmind/tapnet/tree/main/data).

<a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_clustering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Point Clustering"/></a> **Point track based video segmentation**: You can run this colab demo to see how point track based video segmentation works. Given an input video, the point tracks are extracted from TAPIR and further separated into different clusters according to different motion patterns. This is purely based on the low level motion and does not depend on any other cues (i.e. semantics). You can also upload your own video and try it.

## Download Checkpoints

`tapnet/checkpoint/` must contain a file checkpoint.npy that's loadable using our NumpyFileCheckpointer. You can download checkpoints here, which should closely match the ones used in the paper.

model|checkpoint|config|backbone|resolution|DAVIS First (AJ)|DAVIS Strided (AJ)|Kinetics First (AJ)|RoboTAP First (AJ)
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
TAP-Net|[Jax](https://storage.googleapis.com/dm-tapnet/checkpoint.npy)|[tapnet_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapnet_config.py)|TSM-ResNet18|256x256|33.0%|38.4%|38.5%|45.1%
TAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt)|[tapir_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_config.py)|ResNet18|256x256|58.5%|63.3%|50.0%|59.6%
Online TAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy)|[causal_tapir_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/causal_tapir_config.py)|ResNet18|256x256|56.2%|58.3%|51.2%|59.1%
BootsTAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt)|[tapir_bootstrap_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_bootstrap_config.py)|ResNet18|256x256|62.4%|67.4%|55.8%|69.2%
Online BootsTAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt)|[tapir_bootstrap_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_bootstrap_config.py)|ResNet18|256x256|59.7%|61.2%|55.1%|69.1

## TAP-Net and TAPIR Training and Inference

### Installation

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
[the jax manual](https://github.com/google/jax#installation)
to install the correct JAX version with CUDA.

### Training

The configuration file is located at: ```./tapnet/configs/tapnet_config.py```.

You can modify it for your need or create your own config file following
the example of ```tapnet_config.py```.

To launch experiment run the command:

```python3 -m tapnet.training.experiment --config ./tapnet/configs/tapnet_config.py```

or

```python3 -m tapnet.training.experiment --config ./tapnet/configs/tapir_config.py```

### Evaluation

You can run evaluation for a particular dataset (i.e. tapvid_davis) using the command:

```bash
python3 -m tapnet.training.experiment \
  --config=./tapnet/configs/tapir_config.py \
  --jaxline_mode=eval_davis_points \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.davis_points_path=/path/to/tapvid_davis.pkl
```

Available eval datasets are listed in `supervised_point_prediction.py`.

### Inference

You can run inference for a particular video (i.e. horsejump-high.mp4) using the command:

```bash
python3 -m tapnet.training.experiment \
  --config=./tapnet/configs/tapnet_config.py \
  --jaxline_mode=eval_inference \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.inference.input_video_path=horsejump-high.mp4 \
  --config.experiment_kwargs.config.inference.output_video_path=result.mp4 \
  --config.experiment_kwargs.config.inference.resize_height=256 \
  --config.experiment_kwargs.config.inference.resize_width=256 \
  --config.experiment_kwargs.config.inference.num_points=20
```

The inference only serves as an example. It will resize the video to 256x256 resolution, sample 20 random query points on the first frame and track these random points in the rest frames.

Note that this uses jaxline for model inference. A more direct way for model inference can be found on the [colab and real-time demos](#tapir-demos).

## Citing this Work

Please use the following bibtex entry to cite our work:

```
@article{doersch2022tap,
  title={Tap-vid: A benchmark for tracking any point in a video},
  author={Doersch, Carl and Gupta, Ankush and Markeeva, Larisa and Recasens, Adria and Smaira, Lucas and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={13610--13626},
  year={2022}
}
```
```
@inproceedings{doersch2023tapir,
  title={Tapir: Tracking any point with per-frame initialization and temporal refinement},
  author={Doersch, Carl and Yang, Yi and Vecerik, Mel and Gokay, Dilara and Gupta, Ankush and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10061--10072},
  year={2023}
}
```
```
@article{vecerik2023robotap,
  title={Robotap: Tracking arbitrary points for few-shot visual imitation},
  author={Vecerik, Mel and Doersch, Carl and Yang, Yi and Davchev, Todor and Aytar, Yusuf and Zhou, Guangyao and Hadsell, Raia and Agapito, Lourdes and Scholz, Jon},
  journal={International Conference on Robotics and Automation},
  year={2024}
}
```
```
@article{doersch2024bootstap,
  title={BootsTAP: Bootstrapped Training for Tracking-Any-Point},
  author={Doersch, Carl and Yang, Yi and Gokay, Dilara and Luc, Pauline and Koppula, Skanda and Gupta, Ankush and Heyward, Joseph and Goroshin, Ross and Carreira, Jo{\~a}o and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2402.00847},
  year={2024}
}
```

## License and Disclaimer

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

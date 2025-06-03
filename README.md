# Tracking Any Point (TAP)

[[`TAP-Vid`](https://tapvid.github.io/)] [[`TAPIR`](https://deepmind-tapir.github.io/)] [[`RoboTAP`](https://robotap.github.io/)] [[`Blog Post`](https://deepmind-tapir.github.io/blogpost.html)] [[`BootsTAP`](https://bootstap.github.io/)] [[`TAPVid-3D`](https://tapvid3d.github.io/)] [[`TAPNext`](https://tap-next.github.io/)] [[`TRAJAN`](https://trajan-paper.github.io)]

https://github.com/google-deepmind/tapnet/assets/4534987/9f66b81a-7efb-48e7-a59c-f5781c35bebc

<!-- disableFinding(LINE_OVER_80) -->
Welcome to the official Google Deepmind repository for Tracking Any Point (TAP), home of the TAP-Vid and TAPVid-3D Datasets, our top-performing TAPIR model, and our RoboTAP extension.

- [TAP-Vid](https://tapvid.github.io) is a benchmark for models that perform this task, with a collection of ground-truth points for both real and synthetic videos.
- [TAPIR](https://deepmind-tapir.github.io) is a two-stage algorithm which employs two stages: 1) a matching stage, which independently locates a suitable candidate point match for the query point on every other frame, and (2) a refinement stage, which updates both the trajectory and query features based on local correlations. The resulting model is fast and surpasses all prior methods by a significant margin on the TAP-Vid benchmark.
- [RoboTAP](https://robotap.github.io) is a system which utilizes TAPIR point tracks to execute robotics manipulation tasks through efficient imitation in the real world. It also includes a dataset with ground-truth points annotated on real robotics manipulation videos.
- [BootsTAP](https://bootstap.github.io) (or Bootstrapped Training for TAP) uses a large dataset of unlabeled, real-world video to improve tracking accuracy. Specifically, the model is trained to give consistent predictions across different spatial transformations and corruptions of the video, as well as different choices of the query points. We apply it to TAPIR to create BootsTAPIR, which is architecturally similar to TAPIR but substantially outperforms it on TAP-Vid.
- [TAPVid-3D](https://tapvid3d.github.io) is a benchmark and set of metrics for models that perform the 3D point tracking task. The benchmark contains 1M+ computed ground-truth trajectories on 4,000+ real-world videos.
- [TAPNext](https://tap-next.github.io) is our latest, most capable, fastest, yet simplest tracker. It formulates the TAP problem as next token prediction and tracks points simply by propagating information through a network. Note that our best TAPNext checkpoint was fine-tuned using the BootsTAP procedure.
- [TRAJAN](https://trajan-paper.github.io) is our first point TRAJectory AutoeNcoder (TRAJAN). TRAJAN conditions on a set of support point trajectories and reconstructs the trajectories of a held out set of query points. The embedding space learned by TRAJAN can be used to compare distributions of videos, to compare motion trajectories in different videos independent of object appearances, and to evaluate the realism and consistency of videos output by generative video models.

This repository contains the following:

- [TAPNext / TAPIR / BootsTAPIR Demos](#demos) for both online **colab demo** and offline **real-time demo** by cloning this repo
- [TRAJAN Demo](#demos) for online **colab demo**
- [TAP-Vid Benchmark](#tap-vid) for both evaluation **dataset** and evaluation **metrics**
- [RoboTAP Benchmark](#roboTAP) for both evaluation **dataset** and point track based clustering code
- [TAPVid-3D Benchmark](#tapvid-3d) for the evaluation **metrics** and sample **evaluation code** for the TAPVid-3D benchmark.
- [Checkpoints](#checkpoints) for TAP-Net (the baseline presented in the TAP-Vid paper), TAPIR and BootsTAPIR **pre-trained** model weights in both **Jax** and **PyTorch**
- [Instructions](#training) for **training** TAP-Net (the baseline presented in the TAP-Vid paper) and TAPIR on Kubric

## Demos

The simplest way to run TAPNext / TAPIR / BootsTAPIR is to use our colab demos online. You can also
clone this repo and run on your own hardware, including a real-time demo.

### Colab Demo

You can run colab demos to see how TAPIR works. You can also upload your own video and try point tracking with TAPIR.
We provide a few colab demos:

<!-- disableFinding(LINE_OVER_80) -->
1. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/tapnext_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="BootsTAPNext"/></a> **BootsTAPNext**: This is the most powerful TAPNext model that runs online (per-frame). This is the BootsTAPNext model reported in the paper.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
2. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapnext_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="BootsTAPNext PyTorch"/></a> **BootsTAPNext PyTorch**: This is the most powerful TAPNext model re-implemented in PyTorch, which contains the exact architecture & weights as the Jax model.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
2. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Offline TAPIR"/></a> **Standard TAPIR**: This is the most powerful TAPIR / BootsTAPIR model that runs on a whole video at once. We mainly report the results of this model in the paper.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
3. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/causal_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Online TAPIR"/></a> **Online TAPIR**: This is the sequential causal TAPIR / BootsTAPIR model that allows for online tracking on points, which can be run in real-time on a GPU platform.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
4. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_rainbow_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TAPIR Rainbow Visualization"/></a> **Rainbow Visualization**: This visualization is used in many of our teaser videos: it does automatic foreground/background segmentation and corrects the tracks for the camera motion, so you can visualize the paths objects take through real space.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
5. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Offline PyTorch TAPIR"/></a> **Standard PyTorch TAPIR**: This is the TAPIR / BootsTAPIR model re-implemented in PyTorch, which contains the exact architecture & weights as the Jax model.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
6. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_causal_tapir_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Online PyTorch TAPIR"/></a> **Online PyTorch TAPIR**: This is the sequential causal BootsTAPIR model re-implemented in PyTorch, which contains the exact architecture & weights as the Jax model.
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->
7. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/trajan_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="TRAJAN"/></a> **TRAJAN**: This is the point trajectory autoencoder for reconstructing the motion of held out point trajectories conditioned on a set of input point trajectories.
8. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_clustering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Point Clustering"/></a> **RoboTAP**: This is the demo of the segmentation algorithm used in RoboTAP.
9. <a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/kubric_for_tapvid(3d).ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Kubric"/></a> **Kubric for TAPVid(3D)**: This visualization illustrates how to generate groundtruth point tracks in 2D and 3D with world coordinate frame using Kubric dataset.

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
[the jax manual](https://github.com/jax-ml/jax#installation)
to install the correct JAX version with CUDA.

You can then run a pretrained causal TAPIR model on a live camera and select points to track:

```bash
cd ..
python3 ./tapnet/live_demo.py \
```

In our tests, we achieved ~17 fps on 480x480 images on a quadro RTX 4000 (a 2018 mobile GPU).

## Benchmarks

This repository hosts three separate but related benchmarks: TAP-Vid, its later extension RoboTAP, and TAPVid-3D.

### TAP-Vid

https://github.com/google-deepmind/tapnet/assets/4534987/ff5fa5e3-ed37-4480-ad39-42a1e2744d8b

[TAP-Vid](https://tapvid.github.io) is a dataset of videos along with point tracks, either manually annotated or obtained from a simulator. The aim is to evaluate tracking of any trackable point on any solid physical surface. Algorithms receive a single query point on some frame, and must produce the rest of the track, i.e., including where that point has moved to (if visible), and whether it is visible, on every other frame. This requires point-level precision (unlike prior work on box and segment tracking) potentially on deformable surfaces (unlike structure from motion) over the long term (unlike optical flow) on potentially any object (i.e. class-agnostic, unlike prior class-specific keypoint tracking on humans).

More details on downloading, using, and evaluating on the **TAP-Vid benchmark** can be found in the corresponding [README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid).

### RoboTAP

[RoboTAP](https://robotap.github.io/) is a following work of TAP-Vid and TAPIR that demonstrates point tracking models are important for robotics.

The [RoboTAP dataset](https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip) follows the same annotation format as TAP-Vid, but is released as an addition to TAP-Vid. In terms of domain, RoboTAP dataset is mostly similar to TAP-Vid-RGB-Stacking, with a key difference that all robotics videos are real and manually annotated. Video sources and object categories are also more diversified. The benchmark dataset includes 265 videos, serving for evaluation purpose only. More details can be found in the TAP-Vid [README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid). We also provide a [RoboTAP Colab Notebook](https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/tapir_clustering.ipynb) demo of the segmentation algorithm used in the paper.

### TAPVid-3D

TAPVid-3D is a dataset and benchmark for evaluating the task of long-range Tracking Any Point in 3D (TAP-3D).

The benchmark features 4,000+ real-world videos, along with their metric 3D position point trajectories. The dataset is contains three different video sources, and spans a variety of object types, motion patterns, and indoor and outdoor environments. This repository folder contains the code to download and generate these annotations and dataset samples to view. Be aware that it has a separate license from TAP-Vid.

More details on downloading, using, and evaluating on the **TAPVid-3D benchmark** can be found in the corresponding [README](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d).

### A Note on Coordinates

In our storage datasets, (x, y) coordinates are typically in normalized raster coordinates: i.e., (0, 0) is the upper-left corner of the upper-left pixel, and
(1, 1) is the lower-right corner of the lower-right pixel. Our code, however, immediately converts these to regular raster coordinates, matching the output of
the Kubric reader: (0, 0) is the upper-left corner of the upper-left pixel, while (h, w) is the lower-right corner of the lower-right pixel, where h is the
image height in pixels, and w is the respective width.

When working with 2D coordinates, we typically store them in the order (x, y). However, we typically work with 3D coordinates in the order (t, y, x), where
y and x are raster coordinates as above, but t is in frame coordinates, i.e. 0 refers to the first frame, and 0.5 refers to halfway between the first and
second frames. Please take care with this: one pixel error can make a difference according to our metrics.

## Checkpoints

[<img alt="Checkpoints" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" width="200px">](https://huggingface.co/google/tapnet)

`tapnet/checkpoint/` must contain a file checkpoint.npy that's loadable using our NumpyFileCheckpointer. You can download checkpoints below here or on [HuggingFace](https://huggingface.co/google/tapnet), which should closely match the ones used in the paper.
Note: evaluation results in the table are reported under 256x256 inference resolution, but higher resolution can benefit results. For BootsTAPIR, we typically find the best results at 512x512 resolution, and for TAPIR, even higher resolutions than 512x512 can be beneficial.

model|checkpoint|config|backbone|training resolution|DAVIS First (AJ)|DAVIS Strided (AJ)|Kinetics First (AJ)|RoboTAP First (AJ)
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
TAP-Net|[Jax](https://storage.googleapis.com/dm-tapnet/checkpoint.npy)|[tapnet_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapnet_config.py)|TSM-ResNet18|256x256|33.0%|38.4%|38.5%|45.1%
TAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt)|[tapir_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_config.py)|ResNet18|256x256|58.5%|63.3%|50.0%|59.6%
Online TAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy)|[causal_tapir_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/causal_tapir_config.py)|ResNet18|256x256|56.2%|58.3%|51.2%|59.1%
BootsTAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt)|[tapir_bootstrap_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_bootstrap_config.py)|ResNet18 + 4 Convs|256x256 + 512x512|62.4%|67.4%|55.8%|69.2%
Online BootsTAPIR|[Jax](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.npy) & [PyTorch](https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt)|[tapir_bootstrap_config.py](https://github.com/google-deepmind/tapnet/blob/main/configs/tapir_bootstrap_config.py)|ResNet18 + 4 Convs|256x256 + 512x512|59.7%|61.2%|55.1%|69.1%
TAPNext|[Jax](https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz)|[tapnext_demo.ipynb](https://github.com/google-deepmind/tapnet/blob/main/colabs/tapnext_demo.ipynb)|TrecViT-B|256x256|65.25%|68.9%|57.3%|64.1%
TRAJAN|[Jax](https://storage.googleapis.com/dm-tapnet/trajan/track_autoencoder_ckpt.npz)|[trajan_demo.ipynb](https://github.com/google-deepmind/tapnet/blob/main/colabs/trajan_demo.ipynb)

## Training

We provide a Jax training and evaluation framework for TAP-Net and TAPIR in the training directory; see the training [README](https://github.com/google-deepmind/tapnet/tree/main/tapnet/training).

Other researchers have developed a [PyTorch training implementation for TAPIR](https://github.com/riponazad/echotracker), which may be of interest; however, this work is **not** affiliated with Google DeepMind, and its accuracy has not been verified by us.

## Citing this Work

Please use the following bibtex entries to cite our work:

```
@article{doersch2022tap,
  title={{TAP}-Vid: A Benchmark for Tracking Any Point in a Video},
  author={Doersch, Carl and Gupta, Ankush and Markeeva, Larisa and Recasens, Adria and Smaira, Lucas and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={13610--13626},
  year={2022}
}
```
```
@inproceedings{doersch2023tapir,
  title={{TAPIR}: Tracking any point with per-frame initialization and temporal refinement},
  author={Doersch, Carl and Yang, Yi and Vecerik, Mel and Gokay, Dilara and Gupta, Ankush and Aytar, Yusuf and Carreira, Joao and Zisserman, Andrew},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10061--10072},
  year={2023}
}
```
```
@article{vecerik2023robotap,
  title={{RoboTAP}: Tracking arbitrary points for few-shot visual imitation},
  author={Vecerik, Mel and Doersch, Carl and Yang, Yi and Davchev, Todor and Aytar, Yusuf and Zhou, Guangyao and Hadsell, Raia and Agapito, Lourdes and Scholz, Jon},
  journal={International Conference on Robotics and Automation},
  pages={5397--5403},
  year={2024}
}
```
```
@article{doersch2024bootstap,
  title={{BootsTAP}: Bootstrapped Training for Tracking-Any-Point},
  author={Doersch, Carl and Luc, Pauline and Yang, Yi and Gokay, Dilara and Koppula, Skanda and Gupta, Ankush and Heyward, Joseph and Rocco, Ignacio and Goroshin, Ross and Carreira, Jo{\~a}o and Zisserman, Andrew},
  journal={Asian Conference on Computer Vision},
  year={2024}
}
```
```
@article{koppula2024tapvid,
  title={{TAPVid}-{3D}: A Benchmark for Tracking Any Point in {3D}},
  author={Koppula, Skanda and Rocco, Ignacio and Yang, Yi and Heyward, Joe and Carreira, Jo{\~a}o and Zisserman, Andrew and Brostow, Gabriel and Doersch, Carl},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
```
@article{zholus2025tapnext,
  title={TAPNext: Tracking Any Point (TAP) as Next Token Prediction},
  author={Zholus, Artem and Doersch, Carl and Yang, Yi and Koppula, Skanda and Patraucean, Viorica and He, Xu Owen and Rocco, Ignacio and Sajjadi, Mehdi S. M. and Chandar, Sarath and Goroshin, Ross},
  journal={arXiv preprint arXiv:2504.05579},
  year={2025}
}
```
```
@article{allen2025trajan,
  title={Direct Motion Models for Assessing Generated Videos},
  author={Allen, Kelsey and Doersch, Carl and Zhou, Guangyao and Suhail, Mohammed and Driess, Danny and Rocco, Ignacio and Rubanova, Yulia and Kipf, Thomas and Sajjadi, Mehdi S. M. and Murphy, Kevin and Carreira, Joao and van Steenkiste, Sjoerd},
  journal={arXiv preprint},
  year={2025}
}
```

## License and Disclaimer

Copyright 2022-2024 Google LLC

Software and other materials specific to the TAPVid-3D benchmark are covered by the license outlined in tapvid3d/LICENSE file.

All other software in this repository is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at:

https://www.apache.org/licenses/LICENSE-2.0

All other non-software materials released here for the TAP-Vid datasets, i.e. the TAP-Vid annotations, as well as the RGB-Stacking videos and RoboTAP videos, are released under a [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode .

The original source videos of DAVIS come from the val set, and are also licensed under creative commons licenses per their creators; see the [DAVIS dataset](https://davischallenge.org/davis2017/code.html) for details. Kinetics videos are publicly available on YouTube, but subject to their own individual licenses. See the [Kinetics dataset webpage](https://www.deepmind.com/open-source/kinetics) for details.

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

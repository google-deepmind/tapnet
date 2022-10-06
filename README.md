# tapnet

Generic motion understanding from video involves not only
tracking objects, but also perceiving how their surfaces
deform and move. Though this information is necessary to
infer shape and physical interactions, the problem of tracking
arbitrary physical points on surfaces over the long term
has received surprisingly little attention. In this paper,
we first formalize the problem, which we call tracking any
point (TAP), and introduce a companion benchmark: TAP-Vid,
that is composed of real-world videos with accurate
annotations and is made possible by a novel semi-automatic
crowdsourced procedure. We also propose a point tracking
model TAP-Net and show how to train it using a combination
of sim2real and self-supervised learning. TAP-Net outperforms
both optical flow and structure-from-motion methods on the
TAP-Vid benchmark while achieving state-of-the-art performance
on unsupervised human keypoint tracking on JHMDB, even though
the model tracks points on clothes and skin rather than the
joints as intended by the benchmark.

## Installation

Install ffmpeg on your machine:

```sudo apt update```

```sudo apt install ffmpeg```

Install OpenEXR:

```sudo apt-get install libopenexr-dev```

Clone the repository:

```git clone https://github.com/deepmind/tapnet.git```

Add a path to directory where TapNet stored to ```PYTHONPATH```:

```export PYTHONPATH=path/to/diretory/where/tapnet/stored:$PYTHONPATH```

Switch to the project directory:

```cd tapnet```

Install submodules:

```git submodule update --init --recursive```

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

```python ./tapnet/experiment.py --config ./tapnet/configs/tapnet_config.py```

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

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

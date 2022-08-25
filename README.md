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

Clone the repository:

```git clone https://github.com/deepmind/tapnet.git```

Install submodules:

```git submodule update --init --recursive```

Install requirements:

```pip install -r requirements.txt```

## Usage

The configuration file is located at: ```./configs/tapnet_config.py```.

You can modify it for your need or create your own config file following
the example of ```tapnet_config.py```.

To launch experiment run the command:

```python ./tapnet/experiment.py --config ./tapnet/configs/tapnet_config.py```

## Troubleshooting
If you got an exception like:

```No 'tapnet' module.``` or ```No 'kubric' module.``` than
add the project directory to ```PATH``` and ```PYTHONPATH```.

## Citing this work

TBD.

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

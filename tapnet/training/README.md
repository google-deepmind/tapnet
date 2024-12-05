# TAP training setup

This directory contains a reference implementation for training and evaluating TAP-Net and TAPIR using Kubric, following our papers.

## Installation

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
[the jax manual](https://github.com/jax-ml/jax#installation)
to install the correct JAX version with CUDA.

## Training

The configuration file is located at: ```./tapnet/configs/tapnet_config.py```.

You can modify it for your need or create your own config file following
the example of ```tapnet_config.py```.

To launch experiment run the command:

```python3 -m tapnet.training.experiment --config ./tapnet/configs/tapnet_config.py```

or

```python3 -m tapnet.training.experiment --config ./tapnet/configs/tapir_config.py```

## Evaluation

You can run evaluation for a particular dataset (i.e. tapvid_davis) using the command:

```bash
python3 -m tapnet.training.experiment \
  --config=./tapnet/configs/tapir_config.py \
  --jaxline_mode=eval_davis_points \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.davis_points_path=/path/to/tapvid_davis.pkl
```

Available eval datasets are listed in `supervised_point_prediction.py`.

## Inference

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

Note that this uses jaxline for model inference if you are training your own model. A more direct way for model inference can be found on the [colab and real-time demos](#tapir-demos).

# TAPNext++ — VOTSp2026

VOTSp2026 point tracking submission based on [TAPNext++](https://arxiv.org/abs/2604.10582) \[Jung et al., CVPR 2026 - Findings\].

## Approach

TAPNext++ is a recurrent, online point tracker: given a set of query points on a reference frame, it tracks them forward frame-by-frame using a joint video–point attention mechanism. No future frames are accessed — tracking is strictly causal.

This submission augments each real query point with **64 local support points** placed in a small grid around it. Support points are co-tracked via the same attention and then discarded; they never appear in output files. The support cluster radius is specified in the model's input-image space (32px in the 512×512 model input), which gives consistent spatial coverage across the dataset's wide range of source resolutions (224–2560px wide). A checkpoint fine-tuned at 512×512 input resolution is used.

**VOTSp2026 result:** d_avg = 0.647 (\delta@1px=0.274, \delta@2px=0.491,
\delta@4px=0.721, \delta@8px=0.842, \delta@16px=0.908), n=50 sequences.

## Installation

```bash
# Create environment
conda create -n tnpp python=3.12
conda activate tnpp

# PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install tapnet directly from GitHub
pip install -q "tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git"

# VOT toolkit and remaining dependencies
pip install vot-toolkit==0.8.1 opencv-python einops
```

## Checkpoint

The checkpoint is downloaded automatically on first run from:
```
https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt
```
It is cached to `~/.cache/tapnextpp/tapnextpp_512.ckpt` and
reused on subsequent runs.

## Running

```bash
# 1. Initialize VOT workspace
vot initialize vots2026/point --workspace /path/to/workspace

# 2. Add the [TNPP] section to the workspace's trackers.ini
cat /path/to/tapnet/tapnet/tapnextpp/votsp2026/trackers.ini >> /path/to/workspace/trackers.ini

# 3. Evaluate
vot evaluate --workspace /path/to/workspace --persist TNPP

# 4. Analyze
vot analysis --workspace /path/to/workspace --format json TNPP

# 5. Pack for submission
vot pack --workspace /path/to/workspace TNPP
```

The resulting zip must have `identifier: TNPP` in its `manifest.yml` to match
the challenge registration.

### trackers.ini

Because `tapnet` is installed as a package, no `paths` override is needed:

```text
[TNPP]
command = tapnet.tapnextpp.votsp2026.tracker
protocol = folderpython
```

A ready-to-use `trackers.ini` with these settings is included in this directory.

## Folder protocol

The VOT toolkit runs the tracker subprocess once per sequence. For each run it
writes a temporary directory containing:

- `frames_color.txt` — absolute paths to video frames, one per line
- `query_{id}.txt` — frame offset (line 1) and initial point `x,y` (line 2)

The tracker writes:

- `output_{id}.txt` — one line per frame: `0` before query frame, then `x,y`
- `output_{id}_visible.txt` — `1` (visible) or `0` (occluded) per frame

## Module structure

| File | Purpose |
|------|---------|
| `tracker.py` | VOT protocol runner — entry point |
| `model.py` | `TAPNextPP` high-level wrapper (BGR frames in, display-space coords out) |
| `utils.py` | Preprocessing and coordinate-transform utilities |

`TAPNextPP` can also be used standalone outside the VOT context:

```python
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

model = TAPNextPP.from_checkpoint("tapnextpp_512.ckpt", device="cuda", input_resolution=512)
positions, visible, state = model.track_frame(frame_bgr, query_points_xy=np.array([[x, y]]))
for frame in subsequent_frames:
    positions, visible, state = model.track_frame(frame, state=state)
```

## Citation

```text
@InProceedings{Jung_2026_CVPR,
  author    = {Jung, Sebastian and Zholus, Artem and Sundermeyer, Martin and Doersch, Carl
               and Goroshin, Ross and Tan, David Joseph and Chandar, Sarath
               and Triebel, Rudolph and Tombari, Federico},
  title     = {TAPNext++: What's Next for Tracking Any Point (TAP)?},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026},
  pages     = {8429-8438}
}
```

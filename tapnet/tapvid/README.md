# TAP-Vid: A Benchmark for Tracking Any Point in a Video

https://github.com/google-deepmind/tapnet/assets/4534987/ff5fa5e3-ed37-4480-ad39-42a1e2744d8b

[TAP-Vid](https://tapvid.github.io) is a dataset of videos along with point tracks, either manually annotated or obtained from a simulator. The aim is to evaluate tracking of any trackable point on any solid physical surface. Algorithms receive a single query point on some frame, and must produce the rest of the track, i.e., including where that point has moved to (if visible), and whether it is visible, on every other frame. This requires point-level precision (unlike prior work on box and segment tracking) potentially on deformable surfaces (unlike structure from motion) over the long term (unlike optical flow) on potentially any object (i.e. class-agnostic, unlike prior class-specific keypoint tracking on humans).

Our full benchmark incorporates 4 datasets: 30 videos from the [DAVIS val set](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip), 1000 videos from the [Kinetics val set](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip), 50 synthetic [Deepmind Robotics videos](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip) for evaluation, and (almost infinite) point track ground truth on the large-scale synthetic [Kubric dataset](https://github.com/google-research/kubric/tree/main/challenges/point_tracking) for training.

We also include a point tracking model TAP-Net, with code to train it on Kubric dataset. TAP-Net outperforms both optical flow and structure-from-motion methods on the TAP-Vid benchmark while achieving state-of-the-art performance on unsupervised human keypoint tracking on JHMDB, even though the model tracks points on clothes and skin rather than the joints as intended by the benchmark.

## Evaluating on TAP-Vid

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

## Comparison of Tracking With and Without Optical Flow

When annotating videos for the TAP-Vid benchmark, we use a track assist algorithm interpolates between the sparse points that the annotators click, since requiring annotators to click every frame is prohibitively expensive.  Specifically, we find tracks which minimize the discrepancy with the optical flow while still connecting the chosen points. Annotators will then check the interpolations and repeat the annotation until they observe no drift.

To validate that this is a better approach than a simple linear interpolation between clicked points, we annotated several DAVIS videos twice and [compare them side by side](https://storage.googleapis.com/dm-tapnet/content/flow_tracker.html), once using the flow-based interpolation, and again using a naive linear interpolation, which simply moves the point at a constant velocity between points.

<a target="_blank" href="https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/optical_flow_track_assist.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Point Track Annotation"/></a> **Flow assist point annotation**: You can run this colab demo to see how point tracks are annotated with optical flow assistance.

## Downloading TAP-Vid-DAVIS and TAP-Vid-RGB-Stacking

The data are contained in pickle files with download links: [TAP-Vid-DAVIS](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip) and [TAP-Vid-RGB-stacking](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip).

For DAVIS, the pickle file contains a dictionary, where each key is a DAVIS video name, and the values are the frames (4D uint8 tensor), the points (float32 tensor with 3 axes; the first is point id, the second is time, and the third is x/y), and the occlusions (bool tensor with 2 axis; the first is point id, the second is time). RGB-Stacking is the same format, except there is no video name, so it is a list of these structures rather than a dictionary.

## Downloading and Processing TAP-Vid-Kinetics

The labels are contained in a csv file with download link: [TAP-Vid-Kinetics](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip).

The videos are expected as the raw clips from [Kinetics700-2020](https://github.com/cvdfoundation/kinetics-dataset) validation set and stored in a local folder `<video_root_path>`. The clips should be stored as MP4, following the name pattern `f'{youtube_id}_{start_time_sec:06}_{end_time_sec:06}.mp4'`, e.g. 'abcdefghijk_000010_000020.mp4'.

Clips can be stored in any subfolder within the `<video_root_path>`. The most common pattern is to store it as `<video_root_path>/<action_label>/<clip_name>`.

Once the validation clips have been downloaded, a pickle file containing all the information can be generated using the provided script:

```bash
pip3 install -r requirements.txt
python3 generate_tapvid.py \
  --input_csv_path=<path_to_tapvid_kinetics.csv> \
  --output_base_path=<path_to_pickle_folder> \
  --video_root_path=<path_to_raw_videos_root_folder> \
  --alsologtostderr
```

## Downloading RoboTAP

The data are contained in pickle files with download links: [RoboTAP](https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip).

RoboTAP follows the same annotation format as TAP-Vid-DAVIS and TAP-Vid-RGB-stacking. The pickle file contains a dictionary, where each key is a video name, and the values are the frames (4D uint8 tensor), the points (float32 tensor with 3 axes; the first is point id, the second is time, and the third is x/y), and the occlusions (bool tensor with 2 axis; the first is point id, the second is time).

## Visualizing TAP-Vid and RoboTAP Dataset

We also provide a script generating an MP4 with the points painted on top of the frames. The script will work with any of the pickle files. A random clip is chosen from all the available ones and all the point tracks are painted.

```bash
pip3 install -r requirements.txt
python3 visualize.py \
  --input_path=<path_to_pickle_file.pkl> \
  --output_path=<path_to_output_video.mp4> \
  --alsologtostderr
```

## Exemplar Visualization
For visualization examples, we have the full [TAP-Vid-DAVIS](https://storage.googleapis.com/dm-tapnet/content/davis_ground_truth_v2.html) as well as 10 examples from the synthetic [TAP-Vid-Kubric](https://storage.googleapis.com/dm-tapnet/content/kubric_ground_truth.html) and [TAP-Vid-RGB-Stacking](https://storage.googleapis.com/dm-tapnet/content/rgb_stacking_ground_truth_v2.html) datasets.

## Training with TAP-Vid-Kubric
TAP-Vid-DAVIS, TAP-Vid-RGB-stacking and TAP-Vid-Kinetics are mainly used for evaluation purpose. To train the model, we use [TAP-Vid-Kubric](https://github.com/google-research/kubric/tree/main/challenges/point_tracking).

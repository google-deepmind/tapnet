# TAP-Vid: A Benchmark for Tracking Any Point in a Video

## Downloading TAP-Vid-DAVIS and TAP-Vid-RGB-Stacking

The data are contained in pickle files with download links: [TAP-Vid-DAVIS](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip) and [TAP-Vid-RGB-stacking](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip).

For DAVIS, the pickle file contains a dictionary, where each key is a DAVIS video name, and the values are the frames (4D uint8 tensor), the points (float32 tensor with 3 axes; the first is point id, the second is time, and the third is x/y), and the occlusions (bool tensor with 2 axis; the first is point id, the second is time). RGB-Stacking is the same format, except there is no video name, so it is a list of these structures rather than a dictionary.

## Downloading and Processing TAP-Vid-Kinetics

The labels are contained in a csv file with download link: [TAP-Vid-Kinetics](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip).

The videos are expected as the raw clips from [Kinetics700-2020](https://www.deepmind.com/open-source/kinetics) validation set and stored in a local folder `<video_root_path>`. The clips should be stored as MP4, following the name pattern `f'{youtube_id}_{start_time_sec:06}_{end_time_sec:06}.mp4'`, e.g. 'abcdefghijk_000010_000020.mp4'.

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

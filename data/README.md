# Downloading and Visualizing Tap-Vid

## Downloading DAVIS and RGB-Stacking

The data are contained in pickle files with download links: [DAVIS](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip), and [RGB-stacking](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip).

For DAVIS, the pickile file contains a dictionary, where each key is a DAVIS video name, and the values are the video (4D uint8 tensor), the points (float32 tensor with 3 axes; the first is point id, the second is time, and the third is x/y), and the occlusions (bool tensor with 2 axies; the first is point id, the second is time). RGB-Stacking is the same format, except there's no video name, so it's a list of these structures rather than a dictionary.

## Visualizing DAVIS and RGB-Stacking

The script below will generate an MP4 video with the points painted on top of the frames. A random video clip is chosen and all the point tracks are painted.

```
python3 -m pip install -r requirements.txt
python3 visualize_pickle.py \
  --input_pkl_path=<path_to_the_pickle_file.pkl> \
  --output_path=<path_to_the_output_video.mp4> \
  --alsologtostderr
```

## Downloading Kinetics

The labels are contained in a csv file with download link: [Kinetics](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip).

The videos are expected as the raw clips from Kinetics700-2020 validation set and stored in a local folder `<video_root_path>`. The videos should be stored as MP4, following the name pattern `f'{youtube_id}.mp4'`, e.g. 'abcdefghijk.mp4'.

## Visualizing Kinetics

The script below will generate an MP4 video with the points painted on top of the frames. A random video clip is chosen and all the point tracks are painted.

```bash
python3 -m pip install -r requirements.txt
python3 visualize_kinetics.py \
  --input_csv_path=<path_to_the_csv_file.csv> \
  --input_video_dir=<video_root_path> \
  --output_path=<path_to_the_output_video.mp4> \
  --alsologtostderr
```

# Downloading and processing Tap-Vid-Kinetics

## Downloading the raw videos

We expect the raw clips from Kinetics700-2020 validation set to be downloaded
and stored in a local folder `<video_root_path>`. The clips should be stored as
MP4, following the name pattern
`f'{youtube_id}_{start_time_sec:06}_{end_time_sec:06}.mp4'`, e.g.
'abcdefghijk_000010_000020.mp4'.

Clips can be stored in any subfolder within the `<video_root_path>`. The most
common pattern is to store it as `<video_root_path>/<label_name>/<clip_name>`.

## Processing the clips

Once the validation clips have been downloaded, a pickle file containing all the
information can be generated using the provided script:

```bash
python3 -m pip install -r requirements.txt
python3 generate_tapvid.py \
  --csv_path=<path_to_tapvid_kinetics.csv> \
  --output_base_path=<path_to_output_pickle_folder> \
  --video_root_path=<path_to_raw_videos_root_folder> \
  --alsologtostderr
```

## Visualizing annotations

We also provide a script generating an MP4 with the points painted on top of the
frames. The script will work with any of the pickle files (Kinetics,
Davis or Robotics). A random clip is chosen from all the available ones and all
the point tracks are painted.

```bash
python3 -m pip install -r requirements.txt
python3 visualize.py \
  --input_path=<path_to_the_pickle_file.pkl> \
  --output_path=<path_to_the_output_video.mp4> \
  --alsologtostderr
```

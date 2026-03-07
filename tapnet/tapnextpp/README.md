### Disclaimer: This is not an officially supported Google product.

# TAPNext++

This code accompanies the TAPNext++ research at Google.

It contains:

* Checkpoint fine-tuned on PointOdyssey and Kubric1024 for 1024 frames
* $AJ_{RD}$ metric calculation code
* Augmentations tailored to improve re-detection capabilities

## Download checkpoint

<!-- disableFinding(LINE_OVER_80) -->
We provide a torch checkpoint fine-tuned on PointOdyssey v1.2 and Kubric-1024 on 1024 frame sequences. The model is not trained on DynHumans but achieves comparable results.

| Dataset | Metric | Value |
| :--- | :--- | :---: |
| **PointOdyssey** | $\delta^{avg} \uparrow$ | 51.8 |
| | Survival $\uparrow$ | 71.0 |
| | MTE $\downarrow$ | 14.4 |
| | $AJ_{RD} \uparrow$ | 23.3 |
| **DAVIS** | AJ $\uparrow$ | 65.6 |
| | $\delta^{avg} \uparrow$ | 79.0 |
| | OA $\uparrow$ | 92.0 |
| **RoboTAP** | AJ $\uparrow$ | 61.1 |
| | $\delta^{avg} \uparrow$ | 75.4 |
| | OA $\uparrow$ | 89.0 |
| | $AJ_{RD} \uparrow$ | 52.6 |
| **Kinetics** | AJ $\uparrow$ | 53.9 |
| | $\delta^{avg} \uparrow$ | 68.4 |
| | OA $\uparrow$ | 88.7 |

To download the checkpoint run
```
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
```

## Example Colab

We adapted the original TAPNext colab to use our checkpoint: https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/torch_tapnextpp_demo.ipynb

## Metric calculation

The Re-Detection Average Jaccard Metric can be calculated using `./metrics/aj_rd.py:compute_redetection_metrics`.

## Augmentations

Code to generate Roll and Homography augmentations as well as visualizations for them are provided in `./augmentations/`. These augmentations are a unique aspect of TAPNext++ training and are crucial for improving its re-detection capabilities under camera roll and perspective shifts.


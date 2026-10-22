# COCO8 fine-tuning experiment

This checkpoint was actually trained for five epochs from YOLO11n on CPU. It is a small reproducibility example, not a better production detector. Choose **COCO8 training demo — experimental** in Object Detection to try it after installing the optional model dependencies.

The run used the official [COCO8 detection dataset](https://docs.ultralytics.com/datasets/detect/coco8/): four training and four validation images, 320-pixel inputs, batch size 2, seed 42, AdamW with learning rate 0.0001 and ten frozen layers. The original pretrained model uses COCO data, so these images are not an independent test of generalization.

| Validation metric | Pretrained baseline | Fine-tuned checkpoint |
|---|---:|---:|
| Precision | 0.6388 | 0.2461 |
| Recall | 0.5667 | 0.6667 |
| mAP50 | 0.6637 | 0.6610 |
| mAP50–95 | 0.4335 | 0.4096 |

Validation selected the saved checkpoint. YOLO11n remains the app default. A useful improvement requires a representative labeled dataset, more training and an untouched test split.

`metrics.json` holds measured results and the checkpoint checksum. `epochs.csv` is the training history. `best.pt` contains the trained weights with optimizer state removed and local paths replaced by portable configuration. The checkpoint is derived from Ultralytics weights; retain the included upstream AGPL-3.0 license and review its distribution terms. The app accepts this fixed local checkpoint only, not uploaded model files.

Reproduce from the project root after model setup:

```bash
python -m visionlab.train --demo --epochs 5 --output training/my-run
```

The training command downloads COCO8 from the official Ultralytics assets release. Dataset images are not redistributed in this package. Exact results may vary across library versions and hardware. The source repository ignores `.pt` files by default; the downloadable project includes this checkpoint separately from normal Git staging.

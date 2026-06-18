# Cricket Ball Dataset

Training needs images here:

```
dataset/cricket_ball_data/
  train/images/   (+ train/labels/)
  valid/images/   (+ valid/labels/)
```

## Quick setup

From `backend` folder (you are already there if terminal shows `...\backend>`):

```powershell
# Step 1 — download dataset
python scripts/setup_dataset.py --kaggle

# Step 2 — verify
python scripts/setup_dataset.py --check

# Step 3 — train
python scripts/train_improved.py --data data.yaml --epochs 100 --model yolov8m.pt --imgsz 1280 --batch 4
```

## Manual download (no Kaggle CLI)

1. Open: https://www.kaggle.com/datasets/kushagra3204/cricket-ball-dataset-for-yolo
2. Download the zip
3. Extract into `backend/dataset/cricket_ball_data/` so `train/images` and `valid/images` exist
4. Run: `python scripts/setup_dataset.py --check`

## Kaggle API key

1. https://www.kaggle.com/settings -> Create New Token
2. Save `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`
3. Run: `python scripts/setup_dataset.py --kaggle`

## After training

Update `config.yaml`:

```yaml
model:
  path: runs/detect/train_improved/weights/best.pt
```

# Quick Start

## 1. Install Dependencies

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 2. Prepare the Dataset

```powershell
python scripts/prepare_dataset.py --overwrite
```

## 3. Train Your Own Model

Short smoke run:

```powershell
python scripts/train_yolov8.py --epochs 1 --batch 8 --imgsz 320 --device 0 --fraction 0.005 --run-name smoke_fraction_005
```

Formal training example:

```powershell
python scripts/train_yolov8.py --epochs 100 --batch 8 --imgsz 640 --device 0 --workers 2 --run-name grape_yolov8_formal
```

Evaluate a trained model:

```powershell
python scripts/evaluate_yolov8.py --weights artifacts\\models\\grape_yolov8_formal_best.pt --split test --device 0
```

## 4. Use a Public Weight Instead of Training

```powershell
python scripts/public_weights.py --id thesab_grape_leaf_detect_v01 --register --alias grape_public_hf_v1 --default
```

## 5. Run CLI Prediction

```powershell
python scripts/predict_image.py --image "data\\processed\\detection_yolo\\images\\val\\1003_jpg.rf.r0QiwaBju17nzURxTDjY.jpg" --device 0
```

## 6. Start the Desktop GUI

```powershell
python -m grape_disease_net.ui.app
```

## 7. Output Locations

- prediction images: `artifacts/predictions/`
- training logs: `artifacts/logs/`
- imported models: `artifacts/models/library/`
- downloaded public weights: `artifacts/models/public/`
- reports: `artifacts/reports/`


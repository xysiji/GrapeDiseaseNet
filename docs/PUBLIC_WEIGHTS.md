# Public Weights Guide

This project supports both native `ultralytics` YOLO weights and YOLOv5-compatible public weights.

## Recommended Public Candidate

- Model: `thesab/grape-leaf-disease-detector`
- Source: <https://huggingface.co/thesab/grape-leaf-disease-detector>
- Direct weight URL: <https://huggingface.co/thesab/grape-leaf-disease-detector/resolve/main/GrapeLeafDetect_v01.pt?download=true>
- Framework: `YOLOv5`
- Task: `detection`
- Classes:
  - `Grape___Black_rot`
  - `Grape___Esca_(Black_Measles)`
  - `Grape___healthy`
  - `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
- Why it fits:
  - disease classes are close to this project topic
  - weight file is public and directly downloadable
  - the project now includes YOLOv5-compatible inference support

## Public But Not Recommended As Main Model

- Model: `Yasssh2123/Grape`
- Source: <https://huggingface.co/Yasssh2123/Grape>
- Direct weight URL: <https://huggingface.co/Yasssh2123/Grape/resolve/main/best.pt?download=true>
- Reason not recommended:
  - this is a grape segmentation model, not a grape disease detection model

## How To Use Public Weights

### List the built-in catalog

```powershell
python scripts/public_weights.py --list
```

### Download the recommended public grape disease model

```powershell
python scripts/public_weights.py --id thesab_grape_leaf_detect_v01 --download
```

### Download and register it into the project model library

```powershell
python scripts/public_weights.py --id thesab_grape_leaf_detect_v01 --register --alias grape_public_hf_v1 --default
```

After registration, the GUI model dropdown and the CLI predictor can use it directly.


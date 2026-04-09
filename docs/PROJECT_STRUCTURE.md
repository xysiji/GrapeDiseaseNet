# Project Structure

```text
GrapeDiseaseNet/
├─ artifacts/
│  ├─ logs/                # training runs
│  ├─ models/              # local weights, public weights, registry
│  │  ├─ library/          # imported model library
│  │  └─ public/           # downloaded public weight files
│  ├─ predictions/         # prediction images and json outputs
│  └─ reports/             # dataset summaries and training reports
├─ configs/
│  ├─ project.yaml         # main project config
│  └─ public_weights_catalog.json
├─ data/
│  ├─ grape.yolov8/        # original detection dataset
│  ├─ grape.coco/          # original coco export
│  └─ processed/
│     └─ detection_yolo/   # prepared train/val/test dataset
├─ docs/
│  ├─ PROJECT_STRUCTURE.md
│  ├─ PUBLIC_WEIGHTS.md
│  └─ QUICK_START.md
├─ scripts/
│  ├─ check_environment.py
│  ├─ prepare_dataset.py
│  ├─ train_yolov8.py
│  ├─ evaluate_yolov8.py
│  ├─ predict_image.py
│  ├─ register_weights.py
│  └─ public_weights.py
├─ src/
│  └─ grape_disease_net/
│     ├─ common/           # paths, model registry
│     ├─ data/             # dataset preparation logic
│     ├─ training/         # training and evaluation pipeline
│     ├─ inference/        # YOLOv8 + YOLOv5-compatible inference
│     └─ ui/               # PyQt5 MVC desktop app
├─ tests/                  # automated tests
├─ 文件/                    # thesis task documents
├─ README.md
├─ requirements.txt
└─ pyproject.toml
```


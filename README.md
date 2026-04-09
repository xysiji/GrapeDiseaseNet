# 基于深度学习的葡萄叶片病害识别

本项目面向本科毕业设计，目标是完成一个基于 `YOLOv8 + PyQt5` 的葡萄叶片病害检测与识别系统。当前仓库已完成第 1 步基础搭建，后续将继续实现数据预处理、模型训练、推理服务和图形界面。

## 项目目标

- 对葡萄叶片病害图像进行检测与识别
- 使用 YOLOv8 完成病害区域定位与类别判断
- 使用 PyQt5 构建桌面端图形界面
- 按 MVC 思路组织代码，降低模型层与界面层耦合

## 当前数据情况

- `data/grape.yolov8/`：主检测数据集，适合作为 YOLOv8 训练基础
- `data/grape.coco/`：与上面同源的 COCO 格式导出
- `data/Grapes Leaves Dataset (images)-20260401T131841Z-1-001/`：分类数据集，可作为补充分析或对比实验数据

说明：当前检测集目录缺少标准的 `valid/test` 划分，第 2 步会先完成数据检查、重划分与统计。

## 目录结构

```text
GrapeDiseaseNet/
├─ configs/                    # 项目配置
├─ scripts/                    # 可直接运行的脚本入口
├─ src/grape_disease_net/      # 主源码
│  ├─ common/                  # 通用工具
│  ├─ data/                    # 数据处理模块
│  ├─ training/                # 训练相关模块
│  ├─ inference/               # 推理相关模块
│  └─ ui/                      # PyQt5 界面与 MVC 代码
├─ tests/                      # 测试代码
├─ artifacts/                  # 训练结果、日志、预测图等输出
├─ data/                       # 原始与后续处理数据
└─ 文件/                        # 任务书、申报书等毕设文档
```

## 环境建议

- Python `3.10`
- Windows 11
- 建议使用虚拟环境

## 安装方式

先安装 PyTorch，再安装项目依赖：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果你的环境是 CPU 或其他 CUDA 版本，需要把 `torch` 安装命令替换成对应版本。

## 当前可用脚本

```powershell
python scripts/check_environment.py
python scripts/prepare_dataset.py --help
python scripts/train_yolov8.py --help
python scripts/evaluate_yolov8.py --help
python scripts/predict_image.py --help
```

这些脚本已经具备统一入口，数据处理、训练和评估流程可直接通过脚本调度。

## 推理示例

```powershell
python scripts/predict_image.py --image "data\\processed\\detection_yolo\\images\\val\\0_jpg.rf.sxOyoCeQs5E6CqTVFQ91.jpg" --device 0
python scripts/predict_image.py --image-dir "data\\processed\\detection_yolo\\images\\test" --device 0 --limit 10
```

## GUI

```powershell
python -m grape_disease_net.ui.app
```

The GUI supports importing any external `.pt` weights file into the local project model library, then selecting it from the dropdown.

## Import External Weights

```powershell
python scripts/register_weights.py --weights "D:\\your_model\\best.pt" --alias grape_external_v1 --default
```

## Public Weights

```powershell
python scripts/public_weights.py --list
python scripts/public_weights.py --id thesab_grape_leaf_detect_v01 --register --alias grape_public_hf_v1 --default
```

See:

- [docs/QUICK_START.md](docs/QUICK_START.md)
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- [docs/PUBLIC_WEIGHTS.md](docs/PUBLIC_WEIGHTS.md)
- [docs/EXPERIMENT_TEMPLATE.md](docs/EXPERIMENT_TEMPLATE.md)

## 开发路线

1. 项目基础结构搭建
2. 数据集检查、转换与重划分
3. YOLOv8 训练与评估
4. 单图/批量推理
5. PyQt5 图形界面与 MVC 集成
6. 测试、文档与实验结果整理

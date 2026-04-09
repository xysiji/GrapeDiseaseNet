from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.paths import ROOT_DIR, ensure_runtime_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备并重划分葡萄病害检测数据集。")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "project.yaml"),
        help="项目配置文件路径。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不实际处理数据。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，则覆盖重建。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_directories()
    try:
        from grape_disease_net.config import load_config
        from grape_disease_net.data.preparation import prepare_detection_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少运行依赖，请先执行 `pip install -r requirements.txt`。"
        ) from exc

    config = load_config(args.config)
    result = prepare_detection_dataset(
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    summary = result["summary"]
    print("数据集处理完成。")
    print(f"输出目录: {result['output_dir']}")
    print(f"总图像数: {summary['totals']['num_images']}")
    print(f"总标注框数: {summary['totals']['num_boxes']}")
    print("划分结果:")
    for split_name, split_stats in summary["splits"].items():
        print(
            f"  {split_name}: images={split_stats['num_images']}, boxes={split_stats['num_boxes']}"
        )
    if args.dry_run:
        print("当前为 dry-run 模式，未写入文件。")
    else:
        print("已生成 dataset.yaml、manifest 和统计报告。")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

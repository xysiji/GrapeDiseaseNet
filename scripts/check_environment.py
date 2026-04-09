from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.paths import ROOT_DIR, ensure_runtime_directories


def probe_package(name: str) -> str:
    return "installed" if importlib.util.find_spec(name) else "missing"


def main() -> None:
    ensure_runtime_directories()

    print("== Environment Check ==")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {ROOT_DIR}")
    print()

    print("== Package Status ==")
    for package in ("yaml", "cv2", "ultralytics", "PyQt5"):
        print(f"{package}: {probe_package(package)}")
    print()

    if probe_package("yaml") == "installed":
        from grape_disease_net.config import load_config

        config = load_config()
        print("== Config Paths ==")
        for key, value in config["paths"].items():
            resolved = (ROOT_DIR / value).resolve()
            print(f"{key}: {resolved} | exists={resolved.exists()}")
    else:
        print("== Config Paths ==")
        print("PyYAML 未安装，暂时跳过配置文件解析。先执行 `pip install -r requirements.txt`。")


if __name__ == "__main__":
    main()

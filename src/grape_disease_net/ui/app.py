from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from grape_disease_net.ui.controllers.main_controller import MainController
from grape_disease_net.ui.models.inference_model import InferenceViewModel
from grape_disease_net.ui.views.main_window import MainWindow


def launch(config_path: str | Path | None = None) -> int:
    app = QApplication(sys.argv)
    model = InferenceViewModel(config_path=config_path)
    view = MainWindow()
    MainController(model, view)
    view.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(launch())

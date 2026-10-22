"""Capture the fibremodes window into docs/images/gui.png."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab the fibremodes window.")
    parser.add_argument("--wait-ms", type=int, default=4000)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/images/gui.png"))
    args = parser.parse_args()
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    import pyqtgraph as pg
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    pg.setConfigOptions(useOpenGL=False, imageAxisOrder="row-major")

    from fibremodes import FibreEngine, catalog
    from fibremodes.ui import FibreModesWindow

    app = QApplication([])
    window = FibreModesWindow()
    window.spec.kind.setCurrentText("Hermite-Gaussian")
    result = FibreEngine(catalog.specs("HG", mfd=34, group=4, N=128)).solve()
    window.load_result(result)
    window.show()

    def grab() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(args.output))
        app.quit()

    QTimer.singleShot(args.wait_ms, grab)
    app.exec()


if __name__ == "__main__":
    main()

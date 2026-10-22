"""Background solve so the window stays responsive."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..engine import FibreEngine
from ..catalog.schema import SolveSpec


class SolveWorker(QThread):
    """Run ``FibreEngine.solve`` off the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, specs: SolveSpec, parent=None):
        super().__init__(parent)
        self.specs = specs

    def run(self) -> None:
        try:
            self.succeeded.emit(FibreEngine(self.specs).solve())
        except Exception as exc:
            self.failed.emit(str(exc))

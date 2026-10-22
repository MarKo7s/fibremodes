"""Desktop window over the fibre catalog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..engine.modes import ModeSet
from .helpers import make_specs
from .views import ResamplePanel, SpecPanel, Viewport
from .workers import SolveWorker


class FibreModesWindow(QMainWindow):
    """Catalog form, solve, optional resample, and the mode viewport."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fibre modes")
        self.result: ModeSet | None = None
        self._worker: SolveWorker | None = None

        self.spec = SpecPanel()
        self.resample = ResamplePanel()
        self.viewport = Viewport()
        self.viewport.hide()
        self.run_button = QPushButton("Run")
        self.view_button = QPushButton("Viewport")
        self.view_button.setEnabled(False)
        self.status = QLabel("Choose a route and run.")
        self.status.setWordWrap(True)

        self.run_button.clicked.connect(self._run)
        self.view_button.clicked.connect(self._open_viewport)
        self.resample.requested.connect(self._resample)

        controls = QWidget()
        column = QVBoxLayout(controls)
        column.addWidget(self.spec, 1)
        column.addWidget(self.run_button)
        column.addWidget(self.resample)
        buttons = QHBoxLayout()
        buttons.addWidget(self.view_button)
        column.addLayout(buttons)
        column.addWidget(self.status)

        splitter = QSplitter()
        splitter.addWidget(controls)
        splitter.addWidget(self.viewport)
        splitter.setStretchFactor(1, 1)
        self.splitter = splitter
        self.setCentralWidget(splitter)
        self.resize(1100, 720)

    def load_result(self, result: ModeSet) -> None:
        """Store a finished solve and open the viewport on it."""
        self._store(result)
        self._open_viewport()

    def _run(self) -> None:
        try:
            specs = make_specs(
                self.spec.current_key(),
                self.spec.current_solver(),
                self.spec.auto_samples(),
                self.spec.values(),
            )
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.run_button.setEnabled(False)
        self.status.setText("Solving...")
        worker = SolveWorker(specs, self)
        worker.succeeded.connect(self._solved)
        worker.failed.connect(self._failed)
        worker.finished.connect(lambda: self.run_button.setEnabled(True))
        self._worker = worker
        worker.start()

    def _solved(self, result: ModeSet) -> None:
        self._store(result)
        if self.viewport.isVisible():
            self.viewport.set_result(result)

    def _failed(self, message: str) -> None:
        self.status.setText(message)

    def _store(self, result: ModeSet) -> None:
        self.result = result
        self.resample.setEnabled(True)
        self.view_button.setEnabled(True)
        self.status.setText(f"{result.key}: {len(result.modes)} modes")

    def _open_viewport(self) -> None:
        if self.result is None:
            return
        self.viewport.show()
        self.viewport.set_result(self.result)
        self.splitter.setSizes([340, 760])

    def _resample(self) -> None:
        if self.result is None:
            return
        try:
            self.result.resample(self.resample.sample_count(), self.resample.window_max())
        except Exception as exc:
            self.status.setText(str(exc))
            return
        shown = "resampled" if self.result.modes_resampled is not None else "solved"
        self.status.setText(f"{self.result.key}: showing {shown} fields")
        if self.viewport.isVisible():
            self.viewport.set_result(self.result)


def launch():
    """Show the window.

    A bare Python process runs the Qt loop. A notebook kernel is asked to run
    that loop itself, so the window keeps receiving mouse and paint events
    after this function returns.
    """
    import sys

    integrated = _integrate_notebook_qt()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = FibreModesWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    if not integrated:
        app.exec()
    return window


def _integrate_notebook_qt() -> bool:
    """Let a Jupyter kernel pump Qt events. Returns False outside a kernel."""
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False
    shell = get_ipython()
    kernel = getattr(shell, "kernel", None) if shell is not None else None
    if kernel is None:
        return False
    try:
        # PySide6 is already imported by this module, so the kernel keeps that binding.
        shell.run_line_magic("gui", "qt")
    except Exception:
        return False
    return getattr(kernel, "eventloop", None) is not None

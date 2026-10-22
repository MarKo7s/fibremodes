"""Post-solve sample count and window."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class ResamplePanel(QWidget):
    """Sample count and half-width applied by ``ModeSet.resample``."""

    requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = QLineEdit("128")
        self.samples.setPlaceholderText("samples")
        self.window = QLineEdit()
        self.window.setPlaceholderText("half-width, empty keeps the solved window")
        button = QPushButton("Resample")
        button.clicked.connect(self.requested.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Resample"))
        layout.addWidget(self.samples)
        layout.addWidget(self.window, 1)
        layout.addWidget(button)
        self.setEnabled(False)

    def sample_count(self) -> int:
        return int(self.samples.text().strip())

    def window_max(self) -> float | None:
        text = self.window.text().strip()
        if text == "":
            return None
        return float(text)

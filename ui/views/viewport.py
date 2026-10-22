"""Mode slider and the two field images."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer, Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ...engine.modes import ModeSet
from ..helpers import (
    active_axis,
    active_modes,
    circle_radii,
    complex_to_rgb,
    display_scale,
    incoherent_sum,
    intensity_to_rgb,
)

pg.setConfigOptions(useOpenGL=False, imageAxisOrder="row-major")


class Viewport(QWidget):
    """Selected mode, incoherent sum, and a one-second autoplay slider."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: ModeSet | None = None
        self._circles: list[tuple] = []

        self.mode_plot = pg.PlotWidget()
        self.sum_plot = pg.PlotWidget()
        self.mode_image = pg.ImageItem()
        self.sum_image = pg.ImageItem()
        self.mode_plot.addItem(self.mode_image)
        self.sum_plot.addItem(self.sum_image)
        for plot, title in ((self.mode_plot, "Mode"), (self.sum_plot, "Incoherent sum")):
            plot.setAspectLocked(True)
            plot.setTitle(title)
            plot.setBackground("k")
            plot.showGrid(x=False, y=False)

        self.caption = QLabel("No modes")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_index)
        self.auto = QCheckBox("Auto")
        self.auto.toggled.connect(self._toggle_auto)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._advance)

        controls = QHBoxLayout()
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.auto)
        images = QHBoxLayout()
        images.addWidget(self.mode_plot, 1)
        images.addWidget(self.sum_plot, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(images, 1)
        layout.addWidget(self.caption)
        layout.addLayout(controls)

    def set_result(self, result: ModeSet) -> None:
        """Bind the viewer to a solve, using resampled fields when they exist."""
        self._result = result
        count = len(active_modes(result))
        self.slider.blockSignals(True)
        self.slider.setEnabled(count > 0)
        self.slider.setMaximum(max(count - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._draw_sum()
        self._show_index(0)

    def _toggle_auto(self, checked: bool) -> None:
        if checked and self.slider.isEnabled():
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        if self.slider.maximum() <= 0:
            return
        nxt = self.slider.value() + 1
        if nxt > self.slider.maximum():
            nxt = 0
        self.slider.setValue(nxt)

    def _show_index(self, index: int) -> None:
        result = self._result
        modes = [] if result is None else active_modes(result)
        if result is None or not modes:
            self.caption.setText("No modes")
            return
        index = min(max(int(index), 0), len(modes) - 1)
        mode = modes[index]
        self._place_image(self.mode_image, complex_to_rgb(mode.field), result)
        n_eff = "" if mode.n_eff is None else f"   n_eff={mode.n_eff:.5f}"
        self.caption.setText(f"{index + 1}/{len(modes)}   {mode.label}{n_eff}")
        self._draw_circles(result)

    def _draw_sum(self) -> None:
        result = self._result
        if result is None or not active_modes(result):
            return
        self._place_image(self.sum_image, intensity_to_rgb(incoherent_sum(result)), result)

    def _place_image(self, image: pg.ImageItem, rgb: np.ndarray, result: ModeSet) -> None:
        axis = active_axis(result)
        image.setImage(rgb, autoLevels=False)
        if axis is None or len(axis) == 0:
            return
        scale, unit = display_scale(result)
        lo = float(axis[0]) * scale
        hi = float(axis[-1]) * scale
        image.setRect(QRectF(lo, lo, hi - lo, hi - lo))
        for plot in (self.mode_plot, self.sum_plot):
            plot.setLabel("bottom", "x", units=unit)
            plot.setLabel("left", "y", units=unit)
            plot.setRange(xRange=(lo, hi), yRange=(lo, hi), padding=0.02)

    def _draw_circles(self, result: ModeSet) -> None:
        for plot, item in self._circles:
            plot.removeItem(item)
        self._circles = []
        pen = pg.mkPen("w", width=1.5)
        theta = np.linspace(0.0, 2.0 * np.pi, 256)
        for radius in circle_radii(result):
            for plot in (self.mode_plot, self.sum_plot):
                curve = plot.plot(radius * np.cos(theta), radius * np.sin(theta), pen=pen)
                self._circles.append((plot, curve))

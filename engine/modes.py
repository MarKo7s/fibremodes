"""Mode fields returned by ``FibreEngine``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..solvers.fiberprofile.base import FibreProfile


@dataclass
class Mode:
    """One field on the shared output grid.

    ``indices`` names the basis or solver quantum numbers. ``n_eff`` and
    ``beta`` are set for a numerical solve. An analytical basis leaves them
    as ``None``.
    """

    field: np.ndarray
    n_eff: float | None
    beta: float | None
    indices: dict[str, int]
    label: str


@dataclass
class ModeSet:
    """Modes from one ``FibreEngine.solve`` call.

    ``x`` is the Cartesian axis of every ``field``. ``unit`` is ``m`` for a
    numerical profile and ``pixel`` for an analytical basis. ``profile`` is
    the refractive-index object for a numerical solve.
    """

    modes: list[Mode]
    x: np.ndarray | None
    wavelength: float | None
    route: str
    key: str
    unit: str | None
    profile: FibreProfile | None
    modes_resampled: list[Mode] | None = None
    x_resampled: np.ndarray | None = None

    def get_fields(self) -> np.ndarray:
        """Stack the solved mode fields as ``(n_modes, N, N)``."""
        return _stack(self.modes)

    def get_fields_resample(self) -> np.ndarray:
        """Stack the resampled mode fields as ``(n_modes, N, N)``."""
        if self.modes_resampled is None:
            raise RuntimeError("Call resample before get_fields_resample.")
        return _stack(self.modes_resampled)

    def fields(self) -> np.ndarray:
        """Stack the solved mode fields as ``(n_modes, N, N)``."""
        return self.get_fields()

    def display_modes(self) -> list[Mode]:
        """Resampled modes when a resample has been stored, otherwise the solved modes."""
        if self.modes_resampled is not None:
            return self.modes_resampled
        return self.modes

    def display_axis(self) -> np.ndarray | None:
        """Axis that matches ``display_modes``."""
        if self.modes_resampled is not None:
            return self.x_resampled
        return self.x

    def resample(self, n, window_max=None) -> list[Mode]:
        """Interpolate the solved fields onto a new square window.

        ``n`` is the sample count on each side. ``window_max`` is the
        half-width. When it is omitted the half-width is the solved axis.
        Samples outside that axis are zero. The solved ``modes`` stay in
        place. The new fields are kept on ``modes_resampled``.
        """
        if not self.modes or self.x is None:
            raise RuntimeError("There are no solved modes to resample.")
        n_points = int(n)
        if n_points < 2:
            raise ValueError("n must be at least 2.")
        window = float(self.x[-1] if window_max is None else window_max)
        if window <= 0:
            raise ValueError("window_max must be positive.")
        source = np.asarray(self.x, dtype=float)
        axis = np.linspace(-window, window, n_points)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
        resampled = []
        for mode in self.modes:
            field = _interpolate_field(source, np.asarray(mode.field), points, n_points)
            resampled.append(Mode(
                field=field,
                n_eff=mode.n_eff,
                beta=mode.beta,
                indices=dict(mode.indices),
                label=mode.label,
            ))
        self.modes_resampled = resampled
        self.x_resampled = axis
        return resampled


def _stack(modes: list[Mode]) -> np.ndarray:
    if not modes:
        return np.empty((0, 0, 0))
    return np.stack([mode.field for mode in modes], axis=0)


def _interpolate_field(source, field, points, n_points):
    if np.iscomplexobj(field):
        real = _interpolate_real(source, np.real(field), points, n_points)
        imag = _interpolate_real(source, np.imag(field), points, n_points)
        return real + 1j * imag
    return _interpolate_real(source, np.asarray(field, dtype=float), points, n_points)


def _interpolate_real(source, values, points, n_points):
    interpolant = RegularGridInterpolator(
        (source, source), values, bounds_error=False, fill_value=0.0,
    )
    return interpolant(points).reshape(n_points, n_points)

"""Display fields taken from a ``ModeSet``."""

from __future__ import annotations

import numpy as np

from ...engine.modes import ModeSet


def active_modes(result: ModeSet):
    """Resampled modes when present, otherwise the solved modes."""
    return result.display_modes()


def active_axis(result: ModeSet):
    """Axis that matches ``active_modes``."""
    return result.display_axis()


def incoherent_sum(result: ModeSet) -> np.ndarray:
    """Sum of the display-mode intensities ``|E|²``."""
    modes = active_modes(result)
    if not modes:
        return np.zeros((1, 1), dtype=float)
    total = np.zeros(np.asarray(modes[0].field).shape, dtype=float)
    for mode in modes:
        total = total + np.abs(np.asarray(mode.field)) ** 2
    return total


def display_scale(result: ModeSet) -> tuple[float, str]:
    """Scale from the stored axis to the viewport, and the axis unit label."""
    if result.unit == "m":
        return 1e6, "um"
    return 1.0, result.unit or "pixel"


def circle_radii(result: ModeSet) -> list[float]:
    """Boundary radii in viewport units that fall inside the display window."""
    profile = result.profile
    axis = active_axis(result)
    if profile is None or axis is None or len(axis) == 0:
        return []
    scale, _unit = display_scale(result)
    limit = max(abs(float(axis[0])), abs(float(axis[-1]))) * scale
    radii = []
    for radius in profile.boundary_radii():
        drawn = float(radius) * scale
        if 0.0 < drawn <= limit:
            radii.append(drawn)
    return radii

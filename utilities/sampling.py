"""Cartesian axes for resampled mode fields."""

from __future__ import annotations

import numpy as np


def cartesian_axis(n, w_max=None, pixel_pitch=None, default_half_width=None):
    """Square-window axis of ``n`` samples.

    ``w_max`` is the half-width of ``linspace(-w_max, w_max, n)``. When it is
    omitted, ``default_half_width`` is that half-width. ``pixel_pitch``
    instead builds ``arange(-n // 2, n // 2) * pixel_pitch`` for even ``n``,
    which has a sample at 0. An odd ``n`` includes the extra positive index
    ``n // 2`` so the length stays ``n`` and zero is still a sample.
    """
    n_points = int(n)
    if n_points < 2:
        raise ValueError("N must be at least 2.")
    if w_max is not None and pixel_pitch is not None:
        raise ValueError("Pass w_max or pixel_pitch, not both.")
    if pixel_pitch is not None:
        pitch = float(pixel_pitch)
        if pitch <= 0:
            raise ValueError("pixel_pitch must be positive.")
        return (np.arange(n_points) - (n_points // 2)) * pitch
    half = default_half_width if w_max is None else w_max
    if half is None:
        raise ValueError("Pass w_max or pixel_pitch.")
    half = float(half)
    if half <= 0:
        raise ValueError("w_max must be positive.")
    return np.linspace(-half, half, n_points)

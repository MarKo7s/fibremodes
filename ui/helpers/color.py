"""Complex field color used by the viewport."""

from __future__ import annotations

import numpy as np


def complex_to_rgb(field) -> np.ndarray:
    """Map a complex field to an ``uint8`` RGB image.

    Hue is the phase and value is the normalized amplitude, with saturation
    held at one. The result is row-major ``(N, N, 3)``.
    """
    field = np.asarray(field)
    amplitude = np.abs(field)
    peak = max(float(amplitude.max()) if amplitude.size else 0.0, 1e-30)
    hue = (np.angle(field) + np.pi) / (2.0 * np.pi)
    rgb = _hsv_to_rgb(hue, np.ones_like(hue), amplitude / peak)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def intensity_to_rgb(intensity) -> np.ndarray:
    """Map a non-negative intensity image to gray ``uint8`` RGB."""
    values = np.asarray(intensity, dtype=float)
    peak = max(float(values.max()) if values.size else 0.0, 1e-30)
    gray = np.clip(values / peak * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _hsv_to_rgb(hue, saturation, value):
    sector = np.floor(hue * 6.0).astype(int) % 6
    fraction = hue * 6.0 - np.floor(hue * 6.0)
    channel_p = value * (1.0 - saturation)
    channel_q = value * (1.0 - fraction * saturation)
    channel_t = value * (1.0 - (1.0 - fraction) * saturation)
    rgb = np.empty(hue.shape + (3,), dtype=float)
    choices = (
        (value, channel_t, channel_p),
        (channel_q, value, channel_p),
        (channel_p, value, channel_t),
        (channel_p, channel_q, value),
        (channel_t, channel_p, value),
        (value, channel_p, channel_q),
    )
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel, data in enumerate(channels):
            rgb[..., channel][mask] = data[mask]
    return rgb

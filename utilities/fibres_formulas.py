"""Common step-index and graded-index fibre estimates.

``a`` is the core radius. ``a`` and ``wavelength`` must be in the same length
unit, and every mode-field result is returned in that unit. ``V`` is the
normalised frequency. The mode-field diameter is twice the 1/e² intensity
radius ``w0``.
"""

from __future__ import annotations

import numpy as np

# LP11 cutoff of a step-index fibre. The fibre is single-moded for V below this.
LP11_CUTOFF = 2.405


def _positive(name, value):
    value = np.asarray(value, dtype=float)
    if np.any(value <= 0):
        raise ValueError(f"{name} must be positive.")
    return value


def wavenumber(wavelength):
    """Vacuum wavenumber ``k0 = 2π / wavelength``."""
    wavelength = _positive("wavelength", wavelength)
    return 2 * np.pi / wavelength


def numerical_aperture(n_core, n_clad):
    """``NA = sqrt(n_core² - n_clad²)``."""
    n_core = _positive("n_core", n_core)
    n_clad = _positive("n_clad", n_clad)
    if np.any(n_clad >= n_core):
        raise ValueError("n_clad must be smaller than n_core.")
    return np.sqrt(n_core**2 - n_clad**2)


def cladding_index(n_core, NA):
    """``n_clad = sqrt(n_core² - NA²)``."""
    n_core = _positive("n_core", n_core)
    NA = _positive("NA", NA)
    if np.any(NA >= n_core):
        raise ValueError("NA must be smaller than n_core.")
    return np.sqrt(n_core**2 - NA**2)


def relative_index_difference(n_core, n_clad):
    """Approximate profile height ``Δ = (n_core - n_clad) / n_core``."""
    n_core = _positive("n_core", n_core)
    n_clad = _positive("n_clad", n_clad)
    return (n_core - n_clad) / n_core


def relative_index_difference_exact(n_core, n_clad):
    """``Δ = (n_core² - n_clad²) / (2 n_core²)``. This is the Δ inside ``NA = n_core sqrt(2Δ)``."""
    n_core = _positive("n_core", n_core)
    n_clad = _positive("n_clad", n_clad)
    return (n_core**2 - n_clad**2) / (2 * n_core**2)


def numerical_aperture_from_delta(n_core, delta):
    """``NA = n_core sqrt(2Δ)``."""
    n_core = _positive("n_core", n_core)
    delta = _positive("delta", delta)
    return n_core * np.sqrt(2 * delta)


def V_number(a, NA, wavelength):
    """Normalised frequency ``V = 2π a NA / wavelength``.

    The same expression is the step-index ``V`` and the parabolic graded-index
    ``V`` once ``NA`` is ``n_core sqrt(2Δ)``.
    """
    a = _positive("a", a)
    NA = _positive("NA", NA)
    return wavenumber(wavelength) * a * NA


def V_number_from_delta(a, n_core, delta, wavelength):
    """``V = k0 a n_core sqrt(2Δ)``, built from :func:`numerical_aperture_from_delta`."""
    return V_number(a, numerical_aperture_from_delta(n_core, delta), wavelength)


def mode_count_step(V):
    """Spatial-mode count of a multimode step-index fibre, ``V² / 2``."""
    V = _positive("V", V)
    return V**2 / 2


def mode_count_graded(V):
    """Spatial-mode count of a parabolic graded-index fibre, ``V² / 4``."""
    V = _positive("V", V)
    return V**2 / 4


def mode_group_count(V):
    """Highest parabolic mode-group index, about ``V / 2``."""
    V = _positive("V", V)
    return V / 2


def modes_in_groups(n_groups):
    """Spatial modes contained in the first ``n_groups`` groups, ``G(G + 1) / 2``."""
    n_groups = _positive("n_groups", n_groups)
    return n_groups * (n_groups + 1) / 2


def marcuse_ratio(V):
    """Marcuse fit for the step-index fundamental, ``w0 / a``.

    The fit is intended for roughly ``0.8 < V < 2.5``.
    """
    V = _positive("V", V)
    return 0.65 + 1.619 / V**1.5 + 2.879 / V**6


def waist_step_index(a, V):
    """1/e² radius of the step-index fundamental, ``w0 = a`` times :func:`marcuse_ratio`."""
    a = _positive("a", a)
    return a * marcuse_ratio(V)


def MFD_step_index(a, V):
    """Mode-field diameter of the step-index fundamental, ``MFD = 2 w0``."""
    return 2 * waist_step_index(a, V)


def MFD_step_index_from_NA(a, NA, wavelength):
    """Step-index fundamental MFD from radius, NA, and wavelength."""
    return MFD_step_index(a, V_number(a, NA, wavelength))


def waist_graded_index(a, V):
    """1/e² radius of the parabolic fundamental, ``w0 = a sqrt(2 / V)``."""
    a = _positive("a", a)
    V = _positive("V", V)
    return a * np.sqrt(2 / V)


def MFD_graded_index(a, V):
    """Mode-field diameter of the parabolic fundamental.

    ``MFD = 2 w0 = a sqrt(8 / V)``.
    """
    return 2 * waist_graded_index(a, V)


def MFD_graded_index_from_NA(a, NA, wavelength):
    """Parabolic fundamental MFD from radius, NA, and wavelength."""
    return MFD_graded_index(a, V_number(a, NA, wavelength))


def MFD_graded_index_from_delta(a, n_core, delta, wavelength):
    """Parabolic fundamental MFD from radius, core index, Δ, and wavelength."""
    return MFD_graded_index(a, V_number_from_delta(a, n_core, delta, wavelength))

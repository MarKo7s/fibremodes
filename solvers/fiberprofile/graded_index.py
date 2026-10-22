"""Power-law graded-index fibre profile."""

from __future__ import annotations

import numpy as np

from .base import FibreProfile, resolve_cladding
from .utils import resolve_solver_samples
from ...utilities.fibres_formulas import (
    MFD_graded_index,
    V_number_from_delta,
    mode_count_graded,
    mode_count_step,
    numerical_aperture,
)


class GradedIndex(FibreProfile):
    """Alpha-profile graded-index fibre.

    .. math::

        n(r) = n_\\mathrm{core}\\sqrt{1 - 2\\Delta (r/a)^\\alpha}
        \\quad (r \\le a),

        n(r) = n_\\mathrm{clad} \\quad (r > a),

        \\Delta = (n_\\mathrm{core}^2 - n_\\mathrm{clad}^2) / (2 n_\\mathrm{core}^2).

    ``alpha = 2`` is the parabolic profile. Pass ``n_clad`` or ``na``;
    ``n_clad = sqrt(n_core**2 - NA**2)``. ``r_max`` is the outer radius of
    the stored radial curve and must lie outside the core. The equation is
    stored on the internal fine curve. ``n_samples`` is the radial solver
    grid and defaults from ``wavelength`` (metres) when omitted.
    """

    def __init__(
        self,
        core_radius,
        r_max,
        n_core,
        n_clad=None,
        na=None,
        alpha=2.0,
        wavelength=None,
        n_samples=None,
    ):
        self._remember_init(
            core_radius=core_radius,
            r_max=r_max,
            n_core=n_core,
            n_clad=n_clad,
            na=na,
            alpha=alpha,
            wavelength=wavelength,
            n_samples=n_samples,
        )
        self.core_radius = float(core_radius)
        self.n_core = float(n_core)
        self.alpha = float(alpha)
        self.n_clad = resolve_cladding(self.n_core, n_clad=n_clad, na=na)
        if self.core_radius <= 0:
            raise ValueError("core_radius must be positive.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if float(r_max) <= self.core_radius:
            raise ValueError("r_max must be larger than core_radius.")
        self.delta = (self.n_core**2 - self.n_clad**2) / (2.0 * self.n_core**2)
        self.V = None
        self.N_est = None
        aperture = None
        if n_samples is None:
            aperture = na if na is not None else numerical_aperture(self.n_core, self.n_clad)
        n_radial, n_cartesian = resolve_solver_samples(
            wavelength, aperture, r_max, n_samples,
        )
        super().__init__(r_max)
        self.n_radial = n_radial
        self.n_cartesian = n_cartesian

    def estimate_num_modes(self, wav_nm, single_pol = True):
        wav = wav_nm * 1e-9
        self.V = V_number_from_delta(self.core_radius, self.n_core, self.delta, wav)
        div = 2 if single_pol else 1
        if self.alpha == 2.0:
            count = mode_count_graded(self.V)
        else:
            count = mode_count_step(self.V) * self.alpha / (self.alpha + 2.0)
        self.N_est = count / div
        K = int(np.ceil(self.N_est))
        print(f'Estimated number of modes: {K}')
        return K

    def estimate_MFD(self):
        """Parabolic fundamental mode-field diameter, ``a sqrt(8/V)``."""
        self.MFD = MFD_graded_index(self.core_radius, self.V) if self.V is not None else None
        return self.MFD

    def _index_on_radius(self, r):
        inside = r <= self.core_radius
        argument = 1.0 - 2.0 * self.delta * (r / self.core_radius) ** self.alpha
        n = self.n_core * np.sqrt(np.clip(argument, 0.0, None))
        n = np.where(inside, n, self.n_clad)
        return n

    def boundary_radii(self):
        return (self.core_radius,)

    def _annotations(self):
        return (
            (self.core_radius / 2.0, self.n_core, r"$n_{\mathrm{core}}$"),
            ((self.core_radius + self.r_max) / 2.0, self.n_clad, r"$n_{\mathrm{clad}}$"),
        )

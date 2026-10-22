"""Step-index fibre profile."""

from __future__ import annotations

import numpy as np

from .base import FibreProfile, resolve_cladding
from .utils import resolve_solver_samples
from ...utilities.fibres_formulas import MFD_step_index, mode_count_step, numerical_aperture, V_number


class StepIndex(FibreProfile):
    """Step-index fibre.

    .. math::

        n(r) = n_\\mathrm{core} \\quad (r < a), \\qquad
        n(r) = n_\\mathrm{clad} \\quad (r \\ge a).

    Pass ``n_clad`` or ``na``; ``n_clad = sqrt(n_core**2 - NA**2)``.
    ``r_max`` is the outer radius of the stored radial curve and must lie
    outside the core. The equation is stored on the internal fine curve.
    ``n_samples`` is the radial solver grid and defaults from ``wavelength``
    (metres) when omitted.
    """

    def __init__(
        self,
        core_radius,
        r_max,
        n_core,
        n_clad=None,
        na=None,
        wavelength=None,
        n_samples=None,
    ):
        self._remember_init(
            core_radius=core_radius,
            r_max=r_max,
            n_core=n_core,
            n_clad=n_clad,
            na=na,
            wavelength=wavelength,
            n_samples=n_samples,
        )
        self.core_radius = float(core_radius)
        self.n_core = float(n_core)
        self.n_clad = resolve_cladding(self.n_core, n_clad=n_clad, na=na)
        self.V = None
        self.N_est = None
        if self.core_radius <= 0:
            raise ValueError("core_radius must be positive.")
        if float(r_max) <= self.core_radius:
            raise ValueError("r_max must be larger than core_radius.")
        aperture = None
        if n_samples is None:
            aperture = na if na is not None else numerical_aperture(self.n_core, self.n_clad)
        n_radial, n_cartesian = resolve_solver_samples(
            wavelength, aperture, r_max, n_samples,
        )
        super().__init__(r_max, wavelength=wavelength)
        self.n_radial = n_radial
        self.n_cartesian = n_cartesian

    def _index_on_radius(self, r):
        return np.where(r < self.core_radius, self.n_core, self.n_clad)

    def estimate_num_modes(self, wav_nm, single_pol = True):
        wav = wav_nm * 1e-9
        self.V = V_number(self.core_radius, numerical_aperture(self.n_core, self.n_clad), wav)
        div = 2 if single_pol else 1
        self.N_est = mode_count_step(self.V) / div
        K = int(np.ceil(self.N_est))
        print(f'Estimated number of modes: {K}')
        return K

    def estimate_MFD(self):
        """Marcuse mode-field diameter of the fundamental mode."""
        self.MFD = MFD_step_index(self.core_radius, self.V) if self.V is not None else None
        return self.MFD

    def boundary_radii(self):
        return (self.core_radius,)

    def _annotations(self):
        return (
            (self.core_radius / 2.0, self.n_core, r"$n_{\mathrm{core}}$"),
            ((self.core_radius + self.r_max) / 2.0, self.n_clad, r"$n_{\mathrm{clad}}$"),
        )

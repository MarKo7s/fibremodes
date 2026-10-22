"""Double-clad step-index fibre profile."""

from __future__ import annotations

import numpy as np

from .base import FibreProfile, resolve_cladding
from .utils import resolve_solver_samples
from ...utilities.fibres_formulas import (
    MFD_step_index,
    V_number,
    mode_count_step,
    numerical_aperture,
)


class DoubleClad(FibreProfile):
    """Three-level double-clad fibre.

    .. math::

        n_\\mathrm{clad1} = \\sqrt{n_\\mathrm{core}^2 - \\mathrm{NA}_\\mathrm{core}^2},

        n_\\mathrm{clad2} = \\sqrt{n_\\mathrm{clad1}^2 - \\mathrm{NA}_\\mathrm{clad1}^2},

    with :math:`n_\\mathrm{core}` for :math:`r \\le a_\\mathrm{core}`,
    :math:`n_\\mathrm{clad1}` for :math:`a_\\mathrm{core} < r \\le a_\\mathrm{clad1}`,
    and :math:`n_\\mathrm{clad2}` outside that. Pass the two numerical
    apertures, or pass ``n_clad1`` and ``n_clad2`` directly. ``r_max`` must
    lie outside the inner cladding so the outer index is on the stored curve.
    The equation is stored on the internal fine curve. ``n_samples`` is the
    radial solver grid and defaults from ``wavelength`` (metres) when omitted,
    using the larger of the two numerical apertures.
    """

    def __init__(
        self,
        core_radius,
        clad1_radius,
        r_max,
        n_core,
        na_core=None,
        na_clad1=None,
        n_clad1=None,
        n_clad2=None,
        wavelength=None,
        n_samples=None,
    ):
        self._remember_init(
            core_radius=core_radius,
            clad1_radius=clad1_radius,
            r_max=r_max,
            n_core=n_core,
            na_core=na_core,
            na_clad1=na_clad1,
            n_clad1=n_clad1,
            n_clad2=n_clad2,
            wavelength=wavelength,
            n_samples=n_samples,
        )
        self.core_radius = float(core_radius)
        self.clad1_radius = float(clad1_radius)
        self.n_core = float(n_core)
        self.na_core = na_core
        self.na_clad1 = na_clad1

        self.V_core = None
        self.V_clad1 = None
        self.N_est = None

        if self.core_radius <= 0:
            raise ValueError("core_radius must be positive.")
        if self.clad1_radius <= self.core_radius:
            raise ValueError("clad1_radius must be larger than core_radius.")
        if float(r_max) <= self.clad1_radius:
            raise ValueError("r_max must be larger than clad1_radius.")
        if (na_core is None) == (n_clad1 is None):
            raise ValueError("Pass na_core or n_clad1.")
        if na_core is not None:
            self.n_clad1 = resolve_cladding(self.n_core, na=na_core)
        else:
            self.n_clad1 = resolve_cladding(self.n_core, n_clad=n_clad1)
        if (na_clad1 is None) == (n_clad2 is None):
            raise ValueError("Pass na_clad1 or n_clad2.")
        if na_clad1 is not None:
            self.n_clad2 = resolve_cladding(self.n_clad1, na=na_clad1)
        else:
            self.n_clad2 = resolve_cladding(self.n_clad1, n_clad=n_clad2)
        aperture = None
        if n_samples is None:
            core_na = na_core if na_core is not None else numerical_aperture(self.n_core, self.n_clad1)
            clad_na = na_clad1 if na_clad1 is not None else numerical_aperture(self.n_clad1, self.n_clad2)
            aperture = max(float(core_na), float(clad_na))
        n_radial, n_cartesian = resolve_solver_samples(
            wavelength, aperture, r_max, n_samples,
        )
        super().__init__(r_max, wavelength=wavelength)
        self.n_radial = n_radial
        self.n_cartesian = n_cartesian

    def _index_on_radius(self, r):
        n = np.full(r.shape, self.n_clad2, dtype=float)
        n[r <= self.clad1_radius] = self.n_clad1
        n[r <= self.core_radius] = self.n_core
        return n

    def estimate_num_modes(self, wav_nm, single_pol = True):
        wav = wav_nm * 1e-9
        V_core = V_number(self.core_radius, self.na_core, wav)
        V_clad1 = V_number(self.clad1_radius, self.na_clad1, wav)

        div = 2 if single_pol else 1
        N_est = (mode_count_step(V_core) + mode_count_step(V_clad1))/div + 1

        K = int(np.ceil(N_est))
        print(f'Estimated number of modes: {K}')

        self.V_core = V_core
        self.V_clad1 = V_clad1
        self.N_est = N_est

        return K

    def estimate_MFD(self):
        """Estimate the MFD of the fibre. Using Marcuse's for the inner core """        
        self.MFD = MFD_step_index(self.core_radius, self.V_core) if self.V_core is not None else None

        return self.MFD

    def boundary_radii(self):
        return (self.core_radius, self.clad1_radius)

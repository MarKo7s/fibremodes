"""Double-clad step-index fibre profile."""

from __future__ import annotations

import numpy as np

from .base import FibreProfile, resolve_cladding


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
        n_samples=4096,
    ):
        self.core_radius = float(core_radius)
        self.clad1_radius = float(clad1_radius)
        self.n_core = float(n_core)
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
        super().__init__(r_max, n_samples=n_samples)

    def _index_on_radius(self, r):
        n = np.full(r.shape, self.n_clad2, dtype=float)
        n[r <= self.clad1_radius] = self.n_clad1
        n[r <= self.core_radius] = self.n_core
        return n

    def boundary_radii(self):
        return (self.core_radius, self.clad1_radius)

    def _annotations(self):
        return (
            (self.core_radius / 2.0, self.n_core, r"$n_{\mathrm{core}}$"),
            ((self.core_radius + self.clad1_radius) / 2.0, self.n_clad1, r"$n_{\mathrm{clad1}}$"),
            ((self.clad1_radius + self.r_max) / 2.0, self.n_clad2, r"$n_{\mathrm{clad2}}$"),
        )

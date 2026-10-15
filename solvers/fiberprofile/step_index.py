"""Step-index fibre profile."""

from __future__ import annotations

import numpy as np

from .base import FibreProfile, resolve_cladding


class StepIndex(FibreProfile):
    """Step-index fibre.

    .. math::

        n(r) = n_\\mathrm{core} \\quad (r < a), \\qquad
        n(r) = n_\\mathrm{clad} \\quad (r \\ge a).

    Pass ``n_clad`` or ``na``; ``n_clad = sqrt(n_core**2 - NA**2)``.
    ``r_max`` is the outer radius of the stored radial curve and must lie
    outside the core.
    """

    def __init__(self, core_radius, r_max, n_core, n_clad=None, na=None, n_samples=4096):
        self.core_radius = float(core_radius)
        self.n_core = float(n_core)
        self.n_clad = resolve_cladding(self.n_core, n_clad=n_clad, na=na)
        if self.core_radius <= 0:
            raise ValueError("core_radius must be positive.")
        if float(r_max) <= self.core_radius:
            raise ValueError("r_max must be larger than core_radius.")
        super().__init__(r_max, n_samples=n_samples)

    def _index_on_radius(self, r):
        return np.where(r < self.core_radius, self.n_core, self.n_clad)

    def boundary_radii(self):
        return (self.core_radius,)

    def _annotations(self):
        return (
            (self.core_radius / 2.0, self.n_core, r"$n_{\mathrm{core}}$"),
            ((self.core_radius + self.r_max) / 2.0, self.n_clad, r"$n_{\mathrm{clad}}$"),
        )

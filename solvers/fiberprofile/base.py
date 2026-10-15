"""Axisymmetric refractive-index profiles sampled from one stored radial curve."""

from __future__ import annotations

import numpy as np

_INTERNAL_SAMPLES = 4096


def resolve_cladding(n_core, n_clad=None, na=None):
    """Return the cladding index from ``n_clad`` or from ``n_clad = sqrt(n_core**2 - NA**2)``."""
    if (n_clad is None) == (na is None):
        raise ValueError("Pass n_clad or na.")
    n_core = float(n_core)
    if n_core <= 0:
        raise ValueError("n_core must be positive.")
    if na is not None:
        na = float(na)
        if na < 0 or na >= n_core:
            raise ValueError("na must satisfy 0 <= na < n_core.")
        return float(np.sqrt(n_core**2 - na**2))
    n_clad = float(n_clad)
    if n_clad <= 0 or n_clad > n_core:
        raise ValueError("n_clad must satisfy 0 < n_clad <= n_core.")
    return n_clad


class FibreProfile:
    """Refractive-index profile stored as a fine radial curve.

    Subclasses implement ``_index_on_radius``. Construction evaluates that
    equation once. ``index_1d`` and ``index_2d`` interpolate the stored curve,
    so a later sample count does not re-evaluate the profile equation.
    """

    def __init__(self, r_max, n_samples=_INTERNAL_SAMPLES):
        r_max = float(r_max)
        n_samples = int(n_samples)
        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2.")
        self.r_max = r_max
        self._r = np.linspace(0.0, r_max, n_samples)
        self._n = np.asarray(self._index_on_radius(self._r), dtype=float)
        if self._n.shape != self._r.shape:
            raise ValueError("Profile equation must return one index per radius.")
        self._sampled_n = None

    def _index_on_radius(self, r):
        raise NotImplementedError

    def boundary_radii(self) -> tuple[float, ...]:
        """Radii of the index steps, in metres."""
        return ()

    def _annotations(self) -> tuple[tuple[float, float, str], ...]:
        """``(radius, index, label)`` tags drawn on the radial panel."""
        return ()

    def _window(self, window_max):
        if window_max is None:
            return float(self._r[-1])
        window_max = float(window_max)
        if window_max <= 0:
            raise ValueError("window_max must be positive.")
        return window_max

    def index_1d(self, N, window_max=None):
        """Radius and index from ``0`` to ``window_max`` with ``N`` samples.

        ``window_max`` defaults to the stored outer radius. Values past that
        radius keep the outer index. The sample count is remembered for a
        later ``index_2d`` call that omits ``N``.
        """
        N = int(N)
        if N < 2:
            raise ValueError("N must be at least 2.")
        window = self._window(window_max)
        r = np.linspace(0.0, window, N)
        n = np.interp(r, self._r, self._n, left=self._n[0], right=self._n[-1])
        self._sampled_n = N
        return r, n

    def index_2d(self, N=None, window_max=None):
        """Cartesian axis and ``(N, N)`` index on ``[-window_max, window_max]``.

        ``N`` defaults to the sample count of the latest ``index_1d`` call,
        or to the stored radial grid when ``index_1d`` has not been called.
        ``rho = sqrt(x^2 + y^2)`` is interpolated from the stored curve.
        """
        if N is None:
            N = self._sampled_n if self._sampled_n is not None else len(self._r)
        N = int(N)
        if N < 2:
            raise ValueError("N must be at least 2.")
        window = self._window(window_max)
        x = np.linspace(-window, window, N)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        rho = np.sqrt(xx**2 + yy**2)
        n = np.interp(rho, self._r, self._n, left=self._n[0], right=self._n[-1])
        return x, n

    def plot(self, N=None, N_2d=None, window_max=None):
        """Draw the Cartesian index and the radial curve.

        ``N`` is the radial sample count. It defaults to the latest
        ``index_1d`` count, or to the stored grid length. ``N_2d`` defaults
        to that same ``N``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        if N is None:
            N = self._sampled_n if self._sampled_n is not None else len(self._r)
        if N_2d is None:
            N_2d = N
        r, n_r = self.index_1d(N, window_max=window_max)
        x, n_xy = self.index_2d(N=N_2d, window_max=window_max)
        window = float(x[-1])

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(10, 5)
        extent_um = 1e6 * np.array([-window, window, -window, window])
        axes[0].imshow(n_xy, cmap="gray", extent=extent_um, origin="lower")
        axes[0].set_title("Refractive index", fontsize=14)
        axes[0].set_xlabel("x (µm)", fontsize=12)
        axes[0].set_ylabel("y (µm)", fontsize=12)
        axes[0].set_aspect("equal")

        axes[1].plot(r * 1e6, n_r, color="black")
        axes[1].set_title("Refractive index profile", fontsize=14)
        axes[1].set_xlabel("Radius (µm)", fontsize=12)
        axes[1].set_ylabel("Refractive index", fontsize=12)
        axes[1].grid(visible=True, which="both", linestyle="--", linewidth=1)

        n_lo = float(np.min(self._n))
        n_hi = float(np.max(self._n))
        span = max(n_hi - n_lo, 1e-3)
        axes[1].set_ylim(n_lo - 0.2 * span, n_hi + 0.45 * span)
        for radius in self.boundary_radii():
            if radius <= window:
                axes[0].add_patch(Circle(
                    (0, 0), radius * 1e6,
                    edgecolor="white", facecolor="none", linewidth=1.2,
                ))
                axes[1].axvline(radius * 1e6, color="0.4", linestyle="--", linewidth=1)
        for radius, index, label in self._annotations():
            if 0 <= radius <= window:
                axes[1].text(
                    radius * 1e6, index + 0.08 * span, label,
                    ha="center", va="bottom", fontsize=12,
                )
        fig.tight_layout()
        return fig, axes

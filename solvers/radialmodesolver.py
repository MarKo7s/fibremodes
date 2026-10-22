"""Radial finite-difference solver for azimuthally harmonic fibre modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


@dataclass
class RadialMode:
    """One guided radial solution.

    ``R`` is the radial eigenvector on ``solver.r_nodes``, which is the
    input radius with the origin removed. ``M`` is the radial index starting
    at 1. ``mode_group`` is ``2*M + |L| - 1``.
    """

    L: int
    M: int
    beta: float
    mode_group: int
    R: np.ndarray


@dataclass
class CartesianMode:
    """Radial mode interpolated onto a square Cartesian window.

    ``field`` has shape ``(N, N)`` and includes the azimuthal factor
    ``exp(i L phi)``.
    """

    L: int
    M: int
    beta: float
    mode_group: int
    field: np.ndarray


def _solve_azimuthal(r, n_r, L, wavelength, margin, beta_tol):
    """Finite-difference radial Helmholtz solve at one non-negative ``L``.

    The equation is

        d²R/dr² + (1/r) dR/dr + (k0² n(r)² - L²/r²) R = β² R.

    ``r[0]`` is the singular origin and is removed. Even ``L`` uses a Neumann
    condition there and odd ``L`` a Dirichlet condition. The outer edge is
    Dirichlet. A WKB phase integral, plus ``margin``, sets how many
    eigenvalues are requested. Returned columns of ``R`` match ``r[1:]``.
    """
    delta0 = r[1:] - r[:-1]
    delta1 = np.concatenate([(r[2:] - r[1:-1]), [r[-1] - r[-2]]])
    delta2 = delta0 + delta1

    radius = r[1:]
    index = n_r[1:]
    n_nodes = len(radius)
    if n_nodes < 2:
        return np.zeros((0, 0)), np.zeros(0)

    origin_dirichlet = L % 2
    k0 = 2.0 * np.pi / wavelength
    k_squared = (k0 * index) ** 2 - (L / radius) ** 2

    diag_c = -2.0 / (delta1[:n_nodes] * delta0[:n_nodes]) + k_squared
    diag_l = (2.0 / delta0[:n_nodes] - 1.0 / radius) / delta2[:n_nodes]
    diag_r = (2.0 / delta1[:n_nodes] + 1.0 / radius) / delta2[:n_nodes]
    diag_c[0] += (1 - origin_dirichlet) * diag_l[0]
    diag_c[-1] += 0.0 * diag_r[-1]

    operator = diags(
        diagonals=[diag_l[1:], diag_c, diag_r[:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )

    n_min = np.min(index)
    n_max = np.max(index)
    beta_min_sq = (k0 * n_min) ** 2
    beta_max_sq = (k0 * n_max) ** 2
    transverse = k_squared - beta_min_sq
    guided_region = (radius > 0) & (transverse > 0)
    if np.any(guided_region):
        phase_integral = simpson(y=np.sqrt(transverse[guided_region]), x=radius[guided_region])
        estimated_m = int(np.ceil(phase_integral / np.pi + 0.5))
        mode_number_guess = estimated_m + margin
    else:
        mode_number_guess = margin

    solved = eigs(operator, k=int(mode_number_guess), which="LR")
    eigenvals = np.asarray(solved[0], dtype=complex)
    amplitudes = np.asarray(solved[1], dtype=complex)
    guided = np.real(eigenvals) > beta_min_sq
    real_eigenvalue = np.abs(np.imag(eigenvals)) < (beta_tol * np.abs(np.real(eigenvals)))
    below_core = np.real(eigenvals) <= beta_max_sq + beta_tol
    valid = guided & real_eigenvalue & below_core

    eigenvals = eigenvals[valid]
    amplitudes = amplitudes[:, valid]
    beta = np.sqrt(np.real(eigenvals))
    order = np.argsort(beta)[::-1]
    beta = beta[order]
    amplitudes = amplitudes[:, order]
    return amplitudes, beta


class RadialModeSolver:
    """Guided-mode solver for a radially symmetric refractive-index profile.

    Parameters
    ----------
    r, n_r :
        Radius samples starting at 0, and the refractive index on those
        samples. The origin is stored and then dropped inside the
        finite-difference matrix.
    wavelength :
        Vacuum wavelength in the same length unit as ``r``.
    margin :
        Extra eigenvalues requested beyond the WKB estimate at each ``L``.
    beta_tol :
        Tolerance for rejecting complex eigenvalues and eigenvalues above
        ``(k0 n_max)**2``.
    """

    def __init__(self, r, n_r, wavelength, margin=4, beta_tol=1e-8):
        radius = np.asarray(r, dtype=float).reshape(-1)
        index = np.asarray(n_r, dtype=float).reshape(-1)
        if radius.shape != index.shape:
            raise ValueError("r and n_r must have the same length.")
        if radius.size < 4:
            raise ValueError("r needs at least 4 samples, including the origin.")
        if not np.all(np.diff(radius) > 0):
            raise ValueError("r must be strictly increasing.")
        if abs(radius[0]) > 1e-12 * max(radius[-1], 1.0):
            raise ValueError("r must start at 0.")
        if float(wavelength) <= 0:
            raise ValueError("wavelength must be positive.")
        if int(margin) < 0:
            raise ValueError("margin must be non-negative.")
        if float(beta_tol) < 0:
            raise ValueError("beta_tol must be non-negative.")
        self.r = radius.copy()
        self.n_r = index.copy()
        self.r_nodes = self.r[1:].copy()
        self.wavelength = float(wavelength)
        self.margin = int(margin)
        self.beta_tol = float(beta_tol)
        self.modes = []
        self.cartesian_modes = []
        self.x = None
        self._modes_ready = False

    def calculate_modes(self, L_max=None):
        """Solve every guided azimuthal order, or stop after ``L_max``.

        ``L_max is None`` increments ``L`` from 0 until an order has no guided
        eigenvalues. An integer ``L_max`` also stops once ``|L|`` would exceed
        it. Positive ``L`` is stored as ``+L`` and ``-L`` with the same radial
        amplitude. The returned list is sorted by mode group, then by ``L``
        from negative to positive, and saved on ``modes``.
        """
        if L_max is not None:
            L_max = int(L_max)
            if L_max < 0:
                raise ValueError("L_max must be >= 0.")
        modes = []
        azimuthal = 0
        while True:
            if L_max is not None and azimuthal > L_max:
                break
            amplitudes, beta = _solve_azimuthal(
                self.r, self.n_r, azimuthal, self.wavelength, self.margin, self.beta_tol,
            )
            if len(beta) == 0:
                break
            for radial_index, value in enumerate(beta, start=1):
                radial = amplitudes[:, radial_index - 1]
                mode_group = 2 * radial_index + azimuthal - 1
                modes.append(RadialMode(
                    L=azimuthal,
                    M=radial_index,
                    beta=float(value),
                    mode_group=mode_group,
                    R=radial,
                ))
                if azimuthal > 0:
                    modes.append(RadialMode(
                        L=-azimuthal,
                        M=radial_index,
                        beta=float(value),
                        mode_group=mode_group,
                        R=radial,
                    ))
            azimuthal += 1
        modes.sort(key=lambda mode: (mode.mode_group, mode.L))
        self.modes = modes
        self.cartesian_modes = []
        self.x = None
        self._modes_ready = True
        return modes

    def calculate_2d_modes(self, N, window_max=None):
        """Interpolate the stored radial modes onto a square window.

        ``N`` is the sample count on each side. ``window_max`` is the
        half-width, so the axis runs from ``-window_max`` to ``+window_max``.
        When ``window_max`` is omitted the half-width is ``r[-1]``. Calling
        this again with a new ``N`` or ``window_max`` resamples without
        resolving the eigenvalue problem.

        The origin sample is restored before interpolation: ``R(0) = 0`` for
        odd ``L``, otherwise ``R(0) = R[0]``. Outside the solved radius the
        field is zero. The azimuthal factor is ``exp(i L phi)``.
        """
        if not self._modes_ready:
            raise RuntimeError("Call calculate_modes before calculate_2d_modes.")
        n_points = int(N)
        if n_points < 2:
            raise ValueError("N must be at least 2.")
        window = float(self.r[-1] if window_max is None else window_max)
        if window <= 0:
            raise ValueError("window_max must be positive.")
        axis = np.linspace(-window, window, n_points)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        rho = np.sqrt(xx**2 + yy**2)
        phi = np.arctan2(yy, xx)
        radius = np.r_[0.0, self.r_nodes]
        cartesian = []
        for mode in self.modes:
            origin = 0.0 if mode.L % 2 else mode.R[0]
            radial = np.interp(
                rho.ravel(),
                radius,
                np.r_[origin, mode.R],
                right=0.0,
            ).reshape(xx.shape)
            field = radial * np.exp(1j * mode.L * phi)
            cartesian.append(CartesianMode(
                L=mode.L,
                M=mode.M,
                beta=mode.beta,
                mode_group=mode.mode_group,
                field=field,
            ))
        self.cartesian_modes = cartesian
        self.x = axis
        return cartesian

    def modes_2D(self):
        """Stack the Cartesian fields from ``calculate_2d_modes``.

        Returns an array of shape ``(n_modes, N, N)``. Axis 0 is the mode
        index, in the same order as ``cartesian_modes``. The last two axes
        are the ``y`` and ``x`` samples of each ``CartesianMode.field``.
        """
        if not self.cartesian_modes:
            raise RuntimeError("Call calculate_2d_modes before modes_2D.")
        return np.stack([mode.field for mode in self.cartesian_modes], axis=0)

    def get_mfd(self):
        """Mode-field diameter of the fundamental mode.

        The first solved mode is fit to the Gaussian with the same radial
        intensity second moment. The diameter is ``2 w0``, with ``w0`` the
        1/e² intensity radius, in the same length unit as ``r``.
        """
        if not self._modes_ready or len(self.modes) == 0:
            raise RuntimeError("Call calculate_modes before get_mfd.")
        mode = self.modes[0]
        radius = np.r_[0.0, self.r_nodes]
        origin = 0.0 if mode.L % 2 else mode.R[0]
        intensity = np.abs(np.r_[origin, mode.R]) ** 2
        power = np.trapezoid(radius * intensity, radius)
        if power <= 0:
            raise RuntimeError("The fundamental mode has no intensity.")
        second_moment = np.trapezoid(radius**3 * intensity, radius) / power
        return float(2.0 * np.sqrt(2.0 * second_moment))

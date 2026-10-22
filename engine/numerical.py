"""Shared radial and scalar adapter for catalog fibres."""

from __future__ import annotations

import numpy as np

from ..catalog.registry import CatalogEntry
from ..catalog.schema import SolveSpec
from ..solvers.radialmodesolver import RadialModeSolver
from ..solvers.scalarmodesolver import ScalarModeSolver
from .modes import Mode, ModeSet


def solve_numerical(spec: SolveSpec, entry: CatalogEntry) -> ModeSet:
    """Build the profile named by ``entry`` and solve it with ``spec.solver``."""
    profile = _build_profile(spec, entry)
    wavelength = float(spec.parameters["wavelength"])
    margin = int(spec.parameters["margin"])
    grid_n = spec.parameters["grid_n"]
    if grid_n is None:
        grid_n = int(profile.n_cartesian)
    grid_n = int(grid_n)
    window_max = spec.parameters["window_max"]

    if spec.solver == "radial":
        modes, axis = _solve_radial(profile, wavelength, margin, grid_n, window_max)
    else:
        modes, axis = _solve_scalar(profile, wavelength, margin, grid_n, window_max)

    return ModeSet(
        modes=modes,
        x=axis,
        wavelength=wavelength,
        route=spec.route,
        key=spec.key,
        unit="m",
        profile=profile,
    )


def _build_profile(spec: SolveSpec, entry: CatalogEntry):
    if entry.build is None:
        raise RuntimeError(f"Catalog entry {entry.key!r} has no profile constructor.")
    kwargs = {"wavelength": spec.parameters["wavelength"]}
    for param in entry.parameters:
        if param.group != "profile":
            continue
        value = spec.parameters[param.name]
        if value is not None:
            kwargs[param.name] = value
    return entry.build(**kwargs)


def _solve_radial(profile, wavelength, margin, grid_n, window_max):
    radius, index = profile.index_1d()
    solver = RadialModeSolver(radius, index, wavelength, margin=margin)
    solver.calculate_modes()
    cartesian = solver.calculate_2d_modes(grid_n, w_max=window_max)
    k0 = 2.0 * np.pi / wavelength
    modes = []
    for mode in cartesian:
        modes.append(Mode(
            field=mode.field,
            n_eff=float(mode.beta / k0),
            beta=float(mode.beta),
            indices={"L": int(mode.L), "M": int(mode.M), "mode_group": int(mode.mode_group)},
            label=f"L={mode.L}, M={mode.M}",
        ))
    return modes, solver.x


def _solve_scalar(profile, wavelength, margin, grid_n, window_max):
    n_modes = int(profile.estimate_num_modes(wavelength * 1e9))
    axis, index = profile.index_2d()
    solver = ScalarModeSolver(axis, index, wavelength, n_modes, margin=margin)
    solver.calculate_modes()
    resampled = grid_n != int(solver.x.size) or window_max is not None
    if resampled:
        fields = solver.resample_modes(grid_n, w_max=window_max)
        axis = solver.x_resampled
    else:
        fields = solver.modes
        axis = solver.x
    k0 = 2.0 * np.pi / wavelength
    modes = []
    for index_mode, (field, n_eff) in enumerate(zip(fields, solver.n_eff)):
        n_eff = float(n_eff)
        modes.append(Mode(
            field=np.asarray(field),
            n_eff=n_eff,
            beta=n_eff * k0,
            indices={"mode": index_mode},
            label=f"mode {index_mode}",
        ))
    return modes, axis

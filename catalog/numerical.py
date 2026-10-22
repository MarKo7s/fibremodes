"""Refractive-index profiles the numerical route can solve."""

from __future__ import annotations

from ..engine.numerical import solve_numerical
from ..solvers.fiberprofile import DoubleClad, GradedIndex, StepIndex
from .registry import CatalogEntry, catalog
from .schema import Param

_SOLVERS = ("radial", "scalar")


def _solve_params() -> tuple[Param, ...]:
    return (
        Param(
            "solver", str, group="solve", required=True, choices=_SOLVERS,
            help="radial solves n(r). scalar solves the Cartesian index.",
        ),
        Param(
            "wavelength", float, group="solve", required=True, unit="m",
            help="Vacuum wavelength.",
        ),
        Param(
            "margin", int, group="solve", default=4,
            help="Extra eigenvalues requested beyond the estimate. Unguided ones are dropped.",
        ),
    )


def _grid_params() -> tuple[Param, ...]:
    return (
        Param(
            "grid_n", int, group="grid",
            help="Cartesian samples per side. Defaults to the profile Cartesian count.",
        ),
        Param(
            "window_max", float, group="grid", unit="m",
            help="Half-width of the output window. Defaults to the profile outer radius.",
        ),
    )


def _register(key: str, label: str, build, profile_params: tuple[Param, ...]) -> None:
    catalog.register(CatalogEntry(
        key=key,
        route="numerical",
        label=label,
        parameters=profile_params + _solve_params() + _grid_params(),
        solvers=_SOLVERS,
        build=build,
        solve_with=solve_numerical,
    ))


_register(
    "step_index",
    "Step index",
    StepIndex,
    (
        Param("core_radius", float, group="profile", required=True, unit="m", help="Core radius."),
        Param("r_max", float, group="profile", required=True, unit="m", help="Outer radius of the stored index. Must lie outside the core."),
        Param("n_core", float, group="profile", required=True, help="Core refractive index."),
        Param("n_clad", float, group="profile", exclusive_group="cladding", help="Cladding index. Pass this or na."),
        Param("na", float, group="profile", exclusive_group="cladding", help="Numerical aperture. Pass this or n_clad."),
        Param("n_samples", int, group="profile", help="Radial samples for the solver. Omit it to size the grid from wavelength and NA."),
    ),
)

_register(
    "graded_index",
    "Graded index",
    GradedIndex,
    (
        Param("core_radius", float, group="profile", required=True, unit="m", help="Core radius."),
        Param("r_max", float, group="profile", required=True, unit="m", help="Outer radius of the stored index. Must lie outside the core."),
        Param("n_core", float, group="profile", required=True, help="On-axis refractive index."),
        Param("n_clad", float, group="profile", exclusive_group="cladding", help="Cladding index. Pass this or na."),
        Param("na", float, group="profile", exclusive_group="cladding", help="Numerical aperture. Pass this or n_clad."),
        Param("alpha", float, group="profile", default=2.0, help="Power-law exponent. 2 is parabolic."),
        Param("n_samples", int, group="profile", help="Radial samples for the solver. Omit it to size the grid from wavelength and NA."),
    ),
)

_register(
    "double_clad",
    "Double clad",
    DoubleClad,
    (
        Param("core_radius", float, group="profile", required=True, unit="m", help="Core radius."),
        Param("clad1_radius", float, group="profile", required=True, unit="m", help="Inner-cladding radius."),
        Param("r_max", float, group="profile", required=True, unit="m", help="Outer radius of the stored index. Must lie outside the inner cladding."),
        Param("n_core", float, group="profile", required=True, help="Core refractive index."),
        Param("na_core", float, group="profile", exclusive_group="core_cladding", help="Core numerical aperture. Pass this or n_clad1."),
        Param("n_clad1", float, group="profile", exclusive_group="core_cladding", help="Inner-cladding index. Pass this or na_core."),
        Param("na_clad1", float, group="profile", exclusive_group="inner_cladding", help="Inner-cladding numerical aperture. Pass this or n_clad2."),
        Param("n_clad2", float, group="profile", exclusive_group="inner_cladding", help="Outer-cladding index. Pass this or na_clad1."),
        Param("n_samples", int, group="profile", help="Radial samples for the solver. Omit it to size the grid from wavelength and the larger NA."),
    ),
)

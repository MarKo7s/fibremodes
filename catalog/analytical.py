"""Analytical bases the catalog can generate."""

from __future__ import annotations

from ..engine.analytical import solve_hg, solve_lg
from .registry import CatalogEntry, catalog
from .schema import Param


def _register(key: str, label: str, parameters: tuple[Param, ...], solve_with) -> None:
    catalog.register(CatalogEntry(
        key=key,
        route="analytical",
        label=label,
        parameters=parameters,
        solvers=(),
        build=None,
        solve_with=solve_with,
    ))


_register(
    "LG",
    "Laguerre-Gaussian",
    (
        Param("mfd", float, group="basis", required=True, unit="pixel", help="Fundamental mode-field diameter. The generator uses waist mfd/2."),
        Param("group", int, group="basis", required=True, help="Number of mode groups."),
        Param("N", int, group="basis", required=True, help="Samples along each side of the grid."),
        Param("px_size", float, group="basis", required=True, unit="pixel", help="Sample pitch, in the same units as mfd."),
        Param("wholeSet", bool, group="basis", default=False, help="Include the complex-conjugate modes."),
        Param("farfield", bool, group="basis", default=False, help="Apply the far-field Gouy phase."),
        Param("engine", str, group="basis", default="GPU", choices=("GPU", "CPU"), help="LG evaluation backend."),
        Param("multicore", bool, group="basis", default=True, help="Use the parallel CPU or GPU path."),
    ),
    solve_lg,
)

_register(
    "HG",
    "Hermite-Gaussian",
    (
        Param("mfd", float, group="basis", required=True, unit="pixel", help="Fundamental mode-field diameter in pixels. The waist passed to the generator is mfd/2."),
        Param("group", int, group="basis", required=True, help="Number of mode groups. The mode count is group*(group+1)/2."),
        Param("N", int, group="basis", required=True, help="Samples along each side of the grid."),
    ),
    solve_hg,
)

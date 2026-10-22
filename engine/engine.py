"""Facade that solves a catalog spec with the existing mode code."""

from __future__ import annotations

from ..catalog.registry import catalog
from ..catalog.schema import SolveSpec
from .modes import ModeSet


class FibreEngine:
    """Solve one ``SolveSpec`` from ``catalog.specs``.

    The spec selects the catalog entry. That entry calls the existing profile,
    solver, or basis generator. Direct use of those classes is unchanged.
    """

    def __init__(self, specs: SolveSpec):
        if not isinstance(specs, SolveSpec):
            raise TypeError("Pass a SolveSpec from catalog.specs.")
        self.specs = specs

    def solve(self) -> ModeSet:
        """Return the modes for this spec."""
        return catalog.get(self.specs.key).solve(self.specs)

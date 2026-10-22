"""Catalog of analytical bases and numerical fibres."""

from .registry import catalog
from .schema import Param, SolveSpec

# Importing the entry modules registers them on ``catalog``.
from . import analytical as _analytical
from . import numerical as _numerical

__all__ = ["Param", "SolveSpec", "catalog"]

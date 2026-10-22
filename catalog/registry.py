"""Catalog of analytical bases and numerical fibres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .schema import Param, SolveSpec


@dataclass(frozen=True)
class CatalogEntry:
    """One registered basis or fibre.

    ``build`` constructs the existing profile class for a numerical entry.
    ``solve`` is called by ``FibreEngine`` and returns a ``ModeSet``.
    """

    key: str
    route: str
    label: str
    parameters: tuple[Param, ...]
    solvers: tuple[str, ...]
    build: Callable[..., Any] | None
    solve_with: Callable[[SolveSpec, CatalogEntry], Any]

    def solve(self, spec: SolveSpec) -> Any:
        """Run this entry on a spec produced for its key."""
        return self.solve_with(spec, self)

    def __repr__(self) -> str:
        if self.solvers:
            return f"CatalogEntry({self.key!r}, route={self.route!r}, solvers={self.solvers})"
        return f"CatalogEntry({self.key!r}, route={self.route!r})"


class Catalog:
    """Lookup of routes, entries, parameter schemas, and validated specs."""

    def __init__(self) -> None:
        self._entries: dict[str, CatalogEntry] = {}

    def register(self, entry: CatalogEntry) -> None:
        """Add an entry. Importing the catalog package registers the built-in ones."""
        if entry.key in self._entries:
            raise ValueError(f"Catalog key {entry.key!r} is already registered.")
        self._entries[entry.key] = entry

    def routes(self) -> list[str]:
        """Route names in registration order."""
        found: list[str] = []
        for entry in self._entries.values():
            if entry.route not in found:
                found.append(entry.route)
        return found

    def list(self, route: str) -> tuple[CatalogEntry, ...]:
        """Entries for ``analytical`` or ``numerical``."""
        known = self.routes()
        if route not in known:
            names = ", ".join(known) if known else "(empty)"
            raise ValueError(f"Unknown route {route!r}. Known routes: {names}.")
        return tuple(entry for entry in self._entries.values() if entry.route == route)

    def schema(self, key: str) -> tuple[Param, ...]:
        """Parameters accepted by ``specs`` for this key."""
        return self.get(key).parameters

    def get(self, key: str) -> CatalogEntry:
        """Return the entry for ``key``."""
        try:
            return self._entries[key]
        except KeyError:
            known = ", ".join(self._entries) if self._entries else "(empty)"
            raise ValueError(f"Unknown catalog key {key!r}. Known keys: {known}.") from None

    def specs(self, key: str, **kwargs: Any) -> SolveSpec:
        """Validate inputs and return the object passed to ``FibreEngine``."""
        entry = self.get(key)
        known = {param.name: param for param in entry.parameters}
        unknown = sorted(set(kwargs) - set(known))
        if unknown:
            names = ", ".join(unknown)
            accepted = ", ".join(known)
            raise ValueError(
                f"Unknown parameter(s) for {key!r}: {names}. Known parameters: {accepted}."
            )

        solver = _resolve_solver(entry, known.get("solver"), kwargs.get("solver", _MISSING))
        _check_exclusive_groups(entry, kwargs)
        values = _collect_values(entry, known, kwargs, solver)
        return SolveSpec(
            key=entry.key,
            route=entry.route,
            solver=solver,
            parameters=values,
        )


_MISSING = object()


def _resolve_solver(entry: CatalogEntry, param: Param | None, supplied: Any) -> str | None:
    if not entry.solvers:
        return None
    if supplied is _MISSING or supplied is None:
        choices = ", ".join(entry.solvers)
        raise ValueError(f"Pass solver for {entry.key!r}. Choose from: {choices}.")
    if param is None:
        raise ValueError(f"Pass solver for {entry.key!r}.")
    solver = _coerce(param, supplied)
    if solver not in entry.solvers:
        choices = ", ".join(entry.solvers)
        raise ValueError(
            f"Solver {solver!r} is not available for {entry.key!r}. Choose from: {choices}."
        )
    if param.choices is not None and solver not in param.choices:
        choices = ", ".join(str(choice) for choice in param.choices)
        raise ValueError(f"solver must be one of {choices}. Got {solver!r}.")
    return str(solver)


def _check_exclusive_groups(entry: CatalogEntry, kwargs: dict[str, Any]) -> None:
    groups: dict[str, list[Param]] = {}
    for param in entry.parameters:
        if param.exclusive_group:
            groups.setdefault(param.exclusive_group, []).append(param)
    for members in groups.values():
        present = [member.name for member in members if kwargs.get(member.name) is not None]
        if len(present) == 1:
            continue
        names = " or ".join(member.name for member in members)
        got = ", ".join(present) if present else "none"
        raise ValueError(f"Pass exactly one of {names} for {entry.key!r}. Got {got}.")


def _collect_values(
    entry: CatalogEntry,
    known: dict[str, Param],
    kwargs: dict[str, Any],
    solver: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for param in known.values():
        applicable = param.solvers is None or (solver is not None and solver in param.solvers)
        supplied = param.name in kwargs and kwargs[param.name] is not None
        if supplied:
            if not applicable:
                owners = ", ".join(param.solvers or ())
                raise ValueError(
                    f"{param.name} applies to solver(s) {owners}, not {solver!r}."
                )
            value = _coerce(param, kwargs[param.name])
            if param.choices is not None and value not in param.choices:
                choices = ", ".join(str(choice) for choice in param.choices)
                raise ValueError(f"{param.name} must be one of {choices}. Got {value!r}.")
            values[param.name] = value
        elif param.exclusive_group or not applicable:
            values[param.name] = None
        elif param.required:
            raise ValueError(f"Missing required parameter {param.name!r} for {entry.key!r}.")
        else:
            values[param.name] = param.default
    return values


def _coerce(param: Param, value: Any) -> Any:
    if param.value_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{param.name} must be a bool. Got {type(value).__name__}.")
        return value
    if param.value_type is int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{param.name} must be an int. Got {type(value).__name__}.")
        return int(value)
    if param.value_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise ValueError(f"{param.name} must be a float. Got {type(value).__name__}.")
        return float(value)
    if param.value_type is str:
        if not isinstance(value, str):
            raise ValueError(f"{param.name} must be a str. Got {type(value).__name__}.")
        return value
    raise ValueError(f"Unsupported parameter type for {param.name}.")


catalog = Catalog()

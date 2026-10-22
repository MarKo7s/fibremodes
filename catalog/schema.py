"""Parameter descriptions and the spec passed to ``FibreEngine``."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Param:
    """One input on a catalog entry.

    ``exclusive_group`` members are alternatives: a spec must set exactly one
    of them. ``solvers`` limits a parameter to those numerical solvers.
    ``value_type`` is ``bool``, ``int``, ``float``, or ``str``.
    """

    name: str
    value_type: type
    group: str
    required: bool = False
    default: Any = None
    unit: str | None = None
    exclusive_group: str | None = None
    choices: tuple[Any, ...] | None = None
    solvers: tuple[str, ...] | None = None
    help: str = ""

    def __repr__(self) -> str:
        if self.required:
            state = "required"
        else:
            state = f"default={self.default!r}"
        unit = f" [{self.unit}]" if self.unit else ""
        exclusive = f", xor={self.exclusive_group}" if self.exclusive_group else ""
        solvers = f", solvers={','.join(self.solvers)}" if self.solvers else ""
        return f"{self.name}: {self.value_type.__name__}{unit} ({state}{exclusive}{solvers})"


@dataclass(frozen=True)
class SolveSpec:
    """Validated inputs for one catalog entry.

    ``parameters`` includes defaults. Unselected exclusive alternatives are
    ``None``. ``solver`` is set for numerical entries and is ``None`` for an
    analytical basis.
    """

    key: str
    route: str
    solver: str | None
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def __repr__(self) -> str:
        solver = f", solver={self.solver!r}" if self.solver else ""
        return f"SolveSpec({self.key!r}, route={self.route!r}{solver})"

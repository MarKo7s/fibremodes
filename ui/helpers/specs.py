"""Turn the spec form into a catalog ``SolveSpec``."""

from __future__ import annotations

from ...catalog import catalog
from ...catalog.schema import Param, SolveSpec

# Modest values matching the engine example notebook. Lengths are metres.
PREFILL = {
    "HG": {"mfd": "34", "group": "4", "N": "128"},
    "LG": {
        "mfd": "34", "group": "4", "N": "128", "px_size": "1",
        "engine": "CPU", "wholeSet": False, "farfield": False, "multicore": False,
    },
    "step_index": {
        "core_radius": "8e-6", "r_max": "20e-6", "n_core": "1.45", "na": "0.12",
        "wavelength": "1550e-9", "n_samples": "128", "margin": "4",
    },
    "graded_index": {
        "core_radius": "12e-6", "r_max": "24e-6", "n_core": "1.46", "na": "0.15",
        "alpha": "2", "wavelength": "1550e-9", "n_samples": "128", "margin": "4",
    },
    "double_clad": {
        "core_radius": "2e-6", "clad1_radius": "11.5e-6", "r_max": "23e-6", "n_core": "1.45841",
        "na_core": "0.13", "na_clad1": "0.20",
        "wavelength": "843.5e-9", "n_samples": "512", "margin": "4",
    },
}


def prefill(key: str, name: str):
    """Text or bool stored for a form field, or ``None`` when the catalog default is enough."""
    return PREFILL.get(key, {}).get(name)


def make_specs(key: str, solver: str | None, auto_samples: bool, values: dict) -> SolveSpec:
    """Validate the form values. ``grid`` parameters are left for ``ModeSet.resample``."""
    schema = {param.name: param for param in catalog.schema(key)}
    kwargs = {}
    if solver is not None and "solver" in schema:
        kwargs["solver"] = solver
    for name, raw in values.items():
        param = schema.get(name)
        if param is None or param.group == "grid" or name == "solver":
            continue
        if auto_samples and name == "n_samples":
            continue
        if param.solvers and solver not in param.solvers:
            continue
        parsed = _parse(param, raw)
        if parsed is None:
            continue
        kwargs[name] = parsed
    return catalog.specs(key, **kwargs)


def _parse(param: Param, raw):
    if param.value_type is bool:
        return bool(raw)
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if param.value_type is int:
        return int(text)
    if param.value_type is float:
        return float(text)
    if param.value_type is str:
        return text
    raise ValueError(f"Unsupported parameter type for {param.name}.")

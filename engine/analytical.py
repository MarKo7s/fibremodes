"""Adapters that call the existing LG and HG generators."""

from __future__ import annotations

import numpy as np

from ..analytical.HG.fibremodes import makeHGModes
from ..analytical.LG.ModesGen import LGmodes
from ..catalog.registry import CatalogEntry
from ..catalog.schema import SolveSpec
from .modes import Mode, ModeSet


def solve_lg(spec: SolveSpec, entry: CatalogEntry) -> ModeSet:
    """Generate Laguerre-Gaussian modes and label ``p`` and ``l``."""
    params = spec.parameters
    basis = LGmodes(
        mfd=params["mfd"],
        group=params["group"],
        N=params["N"],
        px_size=params["px_size"],
        generateModes=True,
        wholeSet=params["wholeSet"],
        farfield=params["farfield"],
        engine=params["engine"],
        multicore=params["multicore"],
    )
    whole_set = bool(params["wholeSet"])
    if params["farfield"]:
        fields = basis.LGmodesArrayFarField__ if whole_set else basis.LGmodesArrayFarField
    else:
        fields = basis.LGmodesArray__ if whole_set else basis.LGmodesArray
    if fields is None:
        raise RuntimeError("LG mode generation did not return a field array.")
    labels = _lg_labels(np.asarray(basis.index), whole_set)
    if fields.shape[0] != len(labels):
        raise RuntimeError(
            f"LG field count {fields.shape[0]} does not match {len(labels)} index labels."
        )
    modes = [
        Mode(
            field=np.asarray(field),
            n_eff=None,
            beta=None,
            indices={"p": p_index, "l": l_index},
            label=f"LG p={p_index}, l={l_index}",
        )
        for field, (p_index, l_index) in zip(fields, labels)
    ]
    n_points = int(params["N"])
    axis = np.arange(-(n_points // 2), n_points // 2) * float(params["px_size"])
    return _analytical_set(spec, entry, modes, axis)


def solve_hg(spec: SolveSpec, entry: CatalogEntry) -> ModeSet:
    """Generate Hermite-Gaussian modes. The waist is ``mfd / 2``."""
    params = spec.parameters
    n_points = int(params["N"])
    group = int(params["group"])
    fields = np.asarray(makeHGModes(n_points, float(params["mfd"]) / 2.0, group))
    labels = [
        (j, i - j, i)
        for i in range(group)
        for j in range(i + 1)
    ]
    if fields.shape[0] != len(labels):
        raise RuntimeError(
            f"HG field count {fields.shape[0]} does not match {len(labels)} index labels."
        )
    modes = [
        Mode(
            field=np.asarray(field),
            n_eff=None,
            beta=None,
            indices={"m": m_index, "n": n_index, "group": group_index},
            label=f"HG m={m_index}, n={n_index}",
        )
        for field, (m_index, n_index, group_index) in zip(fields, labels)
    ]
    axis = np.arange(-(n_points // 2), n_points // 2, dtype=float)
    return _analytical_set(spec, entry, modes, axis)


def _lg_labels(index: np.ndarray, whole_set: bool) -> list[tuple[int, int]]:
    if index.ndim != 2 or index.shape[0] != 2:
        raise RuntimeError("LG mode index must have shape (2, n_modes).")
    labels: list[tuple[int, int]] = []
    for p_index, l_index in zip(index[0], index[1]):
        p_value = int(p_index)
        l_value = int(l_index)
        labels.append((p_value, l_value))
        if whole_set and l_value != 0:
            labels.append((p_value, -l_value))
    return labels


def _analytical_set(spec: SolveSpec, entry: CatalogEntry, modes: list[Mode], axis: np.ndarray) -> ModeSet:
    return ModeSet(
        modes=modes,
        x=axis,
        wavelength=None,
        route=spec.route,
        key=entry.key,
        unit="pixel",
        profile=None,
    )

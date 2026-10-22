"""Catalog-driven solve form."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...catalog import catalog
from ...catalog.schema import Param
from ..helpers import prefill


class SpecPanel(QWidget):
    """Route, fibre or basis, solver, and the parameters that entry accepts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[str, QWidget] = {}
        self._editors: dict[str, QWidget] = {}

        self.route = QComboBox()
        self.kind = QComboBox()
        self.solver = QComboBox()
        self.auto = QCheckBox("Auto samples")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._apply_auto_samples)

        self._solver_row = _labeled_row("Solver", self.solver)
        self._body = QVBoxLayout()
        body_host = QWidget()
        body_host.setLayout(self._body)

        layout = QVBoxLayout(self)
        layout.addWidget(_labeled_row("Route", self.route))
        layout.addWidget(_labeled_row("Kind", self.kind))
        layout.addWidget(self._solver_row)
        layout.addWidget(self.auto)
        layout.addWidget(body_host)
        layout.addStretch(1)

        self.route.currentIndexChanged.connect(self._rebuild_kinds)
        self.kind.currentIndexChanged.connect(self._rebuild_fields)
        self.solver.currentIndexChanged.connect(self._apply_solver)

        for name in catalog.routes():
            self.route.addItem(name, name)
        self._rebuild_kinds()

    def current_key(self) -> str:
        return str(self.kind.currentData())

    def current_solver(self) -> str | None:
        if self._solver_row.isHidden():
            return None
        text = self.solver.currentText()
        return text or None

    def auto_samples(self) -> bool:
        return self.auto.isChecked()

    def field_names(self) -> list[str]:
        """Parameter names currently built, including hidden rows."""
        names = []
        for name, row in self._rows.items():
            active = getattr(row, "active_name", None)
            names.append(active() if active else name)
        return names

    def visible_names(self) -> list[str]:
        """Parameter names that are shown and would be submitted."""
        names = []
        for name, row in self._rows.items():
            if row.isHidden():
                continue
            active = getattr(row, "active_name", None)
            names.append(active() if active else name)
        return names

    def values(self) -> dict:
        """Raw widget values for the visible parameter rows."""
        collected = {}
        for name in self.visible_names():
            editor = self._editors[name]
            if isinstance(editor, QCheckBox):
                collected[name] = editor.isChecked()
            elif isinstance(editor, QComboBox):
                collected[name] = editor.currentText()
            else:
                collected[name] = editor.text()
        return collected

    def _rebuild_kinds(self) -> None:
        route = self.route.currentData()
        self.kind.blockSignals(True)
        self.kind.clear()
        if route:
            for entry in catalog.list(str(route)):
                self.kind.addItem(entry.label, entry.key)
        self.kind.blockSignals(False)
        self._rebuild_fields()

    def _rebuild_fields(self) -> None:
        while self._body.count():
            item = self._body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()
        self._editors.clear()

        key = self.kind.currentData()
        if not key:
            return
        entry = catalog.get(str(key))
        self._fill_solvers(entry.solvers)
        self.auto.setVisible(any(param.name == "n_samples" for param in entry.parameters))

        seen_groups = set()
        for param in entry.parameters:
            if param.name == "solver" or param.group == "grid":
                continue
            if param.exclusive_group:
                if param.exclusive_group in seen_groups:
                    continue
                seen_groups.add(param.exclusive_group)
                members = [item for item in entry.parameters if item.exclusive_group == param.exclusive_group]
                row, editor = _exclusive_row(str(key), members)
                self._rows[param.exclusive_group] = row
                for member in members:
                    self._editors[member.name] = editor
            else:
                row, editor = _parameter_row(str(key), param)
                self._rows[param.name] = row
                self._editors[param.name] = editor
            self._body.addWidget(row)
        self._apply_solver()
        self._apply_auto_samples()

    def _fill_solvers(self, solvers: tuple[str, ...]) -> None:
        current = self.solver.currentText()
        self.solver.blockSignals(True)
        self.solver.clear()
        for name in solvers:
            self.solver.addItem(name)
        if current in solvers:
            self.solver.setCurrentText(current)
        self._solver_row.setVisible(bool(solvers))
        self.solver.blockSignals(False)

    def _apply_solver(self) -> None:
        solver = self.current_solver()
        key = self.kind.currentData()
        if not key:
            return
        limited = {param.name: param.solvers for param in catalog.schema(str(key)) if param.solvers}
        for name, row in self._rows.items():
            owners = limited.get(name)
            if owners:
                row.setVisible(solver in owners)
        self._apply_auto_samples()

    def _apply_auto_samples(self) -> None:
        row = self._rows.get("n_samples")
        if row is None:
            return
        row.setVisible(not self.auto.isHidden() and not self.auto.isChecked())


def _labeled_row(title: str, widget: QWidget) -> QWidget:
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(title))
    layout.addWidget(widget, 1)
    return host


def _parameter_row(key: str, param: Param) -> tuple[QWidget, QWidget]:
    editor = _editor(key, param)
    host = _labeled_row(_caption(param), editor)
    return host, editor


def _exclusive_row(key: str, members: list[Param]) -> tuple[QWidget, QLineEdit]:
    chosen = _preferred(key, members)
    selector = QComboBox()
    editor = QLineEdit()
    for member in members:
        selector.addItem(member.name, member.name)
    selector.setCurrentText(chosen.name)
    stored = prefill(key, chosen.name)
    editor.setText("" if stored is None else str(stored))
    editor.setPlaceholderText(chosen.unit or "")

    def _use_selected():
        name = str(selector.currentData())
        value = prefill(key, name)
        editor.setText("" if value is None else str(value))

    selector.currentIndexChanged.connect(_use_selected)

    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel("Index"))
    layout.addWidget(selector)
    layout.addWidget(editor, 1)
    host.active_name = lambda: str(selector.currentData())
    return host, editor


def _editor(key: str, param: Param) -> QWidget:
    stored = prefill(key, param.name)
    if param.value_type is bool:
        box = QCheckBox()
        default = param.default if stored is None else stored
        box.setChecked(bool(default))
        return box
    if param.choices:
        combo = QComboBox()
        for choice in param.choices:
            combo.addItem(str(choice))
        selected = param.default if stored is None else stored
        if selected is not None and str(selected) in param.choices:
            combo.setCurrentText(str(selected))
        return combo
    edit = QLineEdit()
    if stored is not None:
        edit.setText(str(stored))
    elif param.default is not None:
        edit.setText(str(param.default))
    edit.setPlaceholderText(param.unit or "")
    return edit


def _preferred(key: str, members: list[Param]) -> Param:
    for member in members:
        if prefill(key, member.name) is not None:
            return member
    return members[0]


def _caption(param: Param) -> str:
    if param.unit:
        return f"{param.name} ({param.unit})"
    return param.name

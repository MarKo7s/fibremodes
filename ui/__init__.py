"""PySide6 window for the fibre catalog. Requires the ``gui`` extra."""

try:
    from .window import FibreModesWindow, launch
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if missing == "pyqtgraph" or missing.startswith("PySide6"):
        raise ModuleNotFoundError(
            'PySide6 and pyqtgraph are required. Install with: pip install "fibremodes[gui]"'
        ) from exc
    raise

__all__ = ["FibreModesWindow", "launch"]

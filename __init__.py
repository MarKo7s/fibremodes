"""Calculate fibre modes — scalar solver and analytical LG bases."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fibremodes")
except PackageNotFoundError:
    __version__ = "1.0.0"

from . import solvers, analytical, utilities

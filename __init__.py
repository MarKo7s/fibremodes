"""Calculate fibre modes — numerical solvers, analytical bases, and FibreEngine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fibremodes")
except PackageNotFoundError:
    __version__ = "1.0.0"

from . import solvers, analytical, utilities
from .catalog import catalog
from .engine import FibreEngine

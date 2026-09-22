"""ChemFAST molecular modeling toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chemfast")
except PackageNotFoundError:
    __version__ = "1.0.0"
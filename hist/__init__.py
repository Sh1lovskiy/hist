"""hist package: unified digital pathology pipeline."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:  # pragma: no cover - defensive packaging helper
    __version__ = version("hist")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

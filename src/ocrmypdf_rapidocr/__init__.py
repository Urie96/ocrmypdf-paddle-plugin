"""RapidOCR (PP-OCRv6) engine plugin for OCRmyPDF."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("ocrmypdf-rapidocr")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

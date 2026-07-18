"""Verisynth: metadata-driven synthetic relational data generation."""

from .engine import Engine
from .metadata import Metadata, MetadataError, load_metadata, parse_metadata
from .scanner import ScanReport, scan_directory

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Metadata",
    "load_metadata",
    "parse_metadata",
    "MetadataError",
    "Engine",
    "ScanReport",
    "scan_directory",
]

"""Verisynth: metadata-driven synthetic relational data generation."""

from .metadata import Metadata, MetadataError, load_metadata, parse_metadata

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Metadata",
    "load_metadata",
    "parse_metadata",
    "MetadataError",
]

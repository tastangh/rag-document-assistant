"""Public processor facade.

Bu dosya bilerek kisa tutulur.
"""

from __future__ import annotations

from .core import DocumentProcessor, process_document_to_markdown
from .document_io import process_directory_to_markdown

__all__ = [
    "DocumentProcessor",
    "process_document_to_markdown",
    "process_directory_to_markdown",
]

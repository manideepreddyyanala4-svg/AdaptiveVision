"""Part alignment against golden references (Milestone M6)."""

from adaptivevision.alignment.model import GoldenReference, LocalizedPart
from adaptivevision.alignment.reference import ReferenceAligner
from adaptivevision.alignment.store import load_golden_reference

__all__ = [
    "GoldenReference",
    "LocalizedPart",
    "ReferenceAligner",
    "load_golden_reference",
]

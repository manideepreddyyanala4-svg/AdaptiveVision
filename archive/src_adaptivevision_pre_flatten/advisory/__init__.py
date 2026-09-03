"""Local-LLM advisory root-cause explanation (Milestone M19)."""

from adaptivevision.advisory.ollama_engine import OllamaAdvisoryEngine
from adaptivevision.advisory.pipeline import advise, build_evidence

__all__ = ["OllamaAdvisoryEngine", "advise", "build_evidence"]

"""Deployment-profile loading and deterministic Pareto recommendation (Milestone M19).

Reads only the JSON artifact written by
``training/benchmark/deployment_export.py`` - production never depends on
training-only code, torch/pandas, or the sweep database directly, only on
this versioned, already-aggregated file. The recommender is pure, small
functions over frozen dataclasses (no LLM involvement, per the architecture
boundary: the advisory layer explains evidence, it never picks a model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class DeploymentProfile:
    """One benchmarked, deployable model configuration.

    Cost/accuracy fields are ``None`` when the corresponding metric was not
    recorded for this configuration (never a fabricated number).

    Attributes:
        model: Method name (for example ``"patchcore"``).
        family: Method family.
        backbone: Backbone/backend identifier.
        config: Configuration key distinguishing variants of ``model``.
        dataset: Dataset this configuration was benchmarked on.
        n_seeds: Number of seeds averaged into this profile.
        benchmark_version: Free-form label identifying the source sweep.
        validated_at: UTC timestamp the profile was exported.
        image_auroc: Mean image-level AUROC.
        pixel_auroc: Mean pixel-level AUROC.
        p50_latency_ms: Mean p50 single-image inference latency.
        p95_latency_ms: Mean p95 single-image inference latency.
        throughput_fps: Mean batch-size-1 throughput.
        model_params_millions: Model parameter count, in millions.
        peak_gpu_memory_mb: Mean peak GPU memory during inference.
        training_wall_clock_seconds: Mean training wall-clock time.
    """

    model: str
    family: str
    backbone: str
    config: str
    dataset: str
    n_seeds: int
    benchmark_version: str
    validated_at: str
    image_auroc: float | None = None
    pixel_auroc: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    throughput_fps: float | None = None
    model_params_millions: float | None = None
    peak_gpu_memory_mb: float | None = None
    training_wall_clock_seconds: float | None = None

    @property
    def label(self) -> str:
        """Return a short human-readable identifier for this profile."""
        return f"{self.model}/{self.config} ({self.dataset})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "model": self.model,
            "family": self.family,
            "backbone": self.backbone,
            "config": self.config,
            "dataset": self.dataset,
            "n_seeds": self.n_seeds,
            "benchmark_version": self.benchmark_version,
            "validated_at": self.validated_at,
            "image_auroc": self.image_auroc,
            "pixel_auroc": self.pixel_auroc,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_fps": self.throughput_fps,
            "model_params_millions": self.model_params_millions,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "training_wall_clock_seconds": self.training_wall_clock_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            model=data["model"],
            family=data["family"],
            backbone=data["backbone"],
            config=data["config"],
            dataset=data["dataset"],
            n_seeds=data["n_seeds"],
            benchmark_version=data["benchmark_version"],
            validated_at=data["validated_at"],
            image_auroc=data.get("image_auroc"),
            pixel_auroc=data.get("pixel_auroc"),
            p50_latency_ms=data.get("p50_latency_ms"),
            p95_latency_ms=data.get("p95_latency_ms"),
            throughput_fps=data.get("throughput_fps"),
            model_params_millions=data.get("model_params_millions"),
            peak_gpu_memory_mb=data.get("peak_gpu_memory_mb"),
            training_wall_clock_seconds=data.get("training_wall_clock_seconds"),
        )


def load_deployment_profiles(path: Path) -> tuple[DeploymentProfile, ...]:
    """Load :class:`DeploymentProfile` records from a JSON artifact.

    Args:
        path: Path to a JSON file holding a list of profile objects, as
            written by ``training/benchmark/deployment_export.py``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` does not contain a JSON list.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"{path} does not contain a JSON list of deployment profiles"
        raise ValueError(msg)
    return tuple(DeploymentProfile.from_dict(item) for item in data)


def pareto_frontier(profiles: Sequence[DeploymentProfile]) -> tuple[DeploymentProfile, ...]:
    """Return the profiles that are Pareto-optimal for (higher AUROC, lower p95 latency).

    A profile missing ``image_auroc`` or ``p95_latency_ms`` is excluded: an
    undefined trade-off cannot be judged optimal or dominated.
    """
    candidates = [p for p in profiles if p.image_auroc is not None and p.p95_latency_ms is not None]
    optimal = []
    for p in candidates:
        dominated = any(
            q is not p
            and q.image_auroc is not None
            and q.p95_latency_ms is not None
            and p.image_auroc is not None
            and p.p95_latency_ms is not None
            and q.image_auroc >= p.image_auroc
            and q.p95_latency_ms <= p.p95_latency_ms
            and (q.image_auroc > p.image_auroc or q.p95_latency_ms < p.p95_latency_ms)
            for q in candidates
        )
        if not dominated:
            optimal.append(p)
    return tuple(optimal)


def feasible_profiles(
    profiles: Sequence[DeploymentProfile],
    *,
    max_latency_ms: float,
    min_auroc: float,
    max_model_size_millions: float | None = None,
) -> tuple[DeploymentProfile, ...]:
    """Return the profiles satisfying every supplied constraint.

    Args:
        profiles: Candidate profiles.
        max_latency_ms: Maximum acceptable ``p95_latency_ms``.
        min_auroc: Minimum acceptable ``image_auroc``.
        max_model_size_millions: Maximum acceptable ``model_params_millions``,
            or ``None`` to not constrain model size.
    """
    return tuple(
        p for p in profiles if _is_feasible(p, max_latency_ms, min_auroc, max_model_size_millions)
    )


def recommend(
    profiles: Sequence[DeploymentProfile],
    *,
    max_latency_ms: float,
    min_auroc: float,
    max_model_size_millions: float | None = None,
) -> DeploymentProfile | None:
    """Deterministically recommend one profile under the given constraints.

    Policy: among the Pareto-optimal profiles meeting every constraint, pick
    the one with the highest ``image_auroc``; ties are broken by the lowest
    ``p95_latency_ms``, then by ``(model, config, dataset)`` so the result is
    fully deterministic. This function never involves an LLM.

    Returns:
        The recommended profile, or ``None`` if no profile satisfies every
        constraint.
    """
    feasible = feasible_profiles(
        profiles,
        max_latency_ms=max_latency_ms,
        min_auroc=min_auroc,
        max_model_size_millions=max_model_size_millions,
    )
    optimal = pareto_frontier(feasible)
    if not optimal:
        return None
    return min(
        optimal,
        key=lambda p: (
            -(p.image_auroc or 0.0),
            p.p95_latency_ms or 0.0,
            p.model,
            p.config,
            p.dataset,
        ),
    )


def explain_recommendation(
    profile: DeploymentProfile,
    *,
    max_latency_ms: float,
    min_auroc: float,
    n_feasible: int,
) -> str:
    """Build a one-line, plain-English reason ``profile`` was recommended."""
    auroc_pct = f"{profile.image_auroc * 100:.1f}%" if profile.image_auroc is not None else "n/a"
    latency = f"{profile.p95_latency_ms:.1f}ms" if profile.p95_latency_ms is not None else "n/a"
    return (
        f"{profile.label}: AUROC={auroc_pct} (>= {min_auroc * 100:.1f}% required), "
        f"p95 latency={latency} (<= {max_latency_ms:.0f}ms budget) - "
        f"highest accuracy among {n_feasible} configs meeting your constraints."
    )


def _is_feasible(
    profile: DeploymentProfile,
    max_latency_ms: float,
    min_auroc: float,
    max_model_size_millions: float | None,
) -> bool:
    """Return ``True`` if ``profile`` satisfies every supplied constraint."""
    if profile.image_auroc is None or profile.image_auroc < min_auroc:
        return False
    if profile.p95_latency_ms is None or profile.p95_latency_ms > max_latency_ms:
        return False
    if max_model_size_millions is not None:
        if profile.model_params_millions is None:
            return False
        if profile.model_params_millions > max_model_size_millions:
            return False
    return True

"""Adapter exposing the Anomalib model zoo through the benchmark interface.

The methods in ``methods_native.py`` are all training-free, which is a
deliberate constraint: they are the ones that stay exportable as a single ONNX
graph and re-fit in minutes on the line. But most of the recent literature --
EfficientAD, Dinomaly, GLASS, Reverse Distillation, the normalizing-flow
family -- needs gradient descent, and reimplementing a dozen papers here would
be a dozen chances to get one subtly wrong and publish a misleading number.

So those are delegated to Anomalib, which already maintains tested
implementations, and this module makes them look identical to a native method
from the runner's point of view: fit on normal images, return per-image
scores. Metrics are then computed by *our* ``evaluation`` module from raw
scores, not read out of Anomalib's own metric objects, so both halves of the
zoo are measured the same way.

Anomalib is an optional dependency. If it is not installed, this module
registers nothing and the native zoo still runs:

    pip install anomalib

Two honest caveats about anything reported from this backend:

* Anomalib models use their own default input geometry, so they do not see the
  per-dataset aspect ratios in ``benchmark.data.DEFAULT_SIZE``. Their numbers
  are comparable to each other and broadly comparable to the native methods,
  but a small gap on the non-square datasets (Kolektor, Severstal) may be
  geometry rather than method.
* They fit by gradient descent, so unlike the native half their results depend
  on epoch count and seed.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmark.data import DatasetConfig
from benchmark.registry import MethodSpec, RunOptions, register

try:  # pragma: no cover - exercised only when the optional dep is present
    import anomalib.models as anomalib_models
    from anomalib.data import Folder
    from anomalib.engine import Engine

    ANOMALIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANOMALIB_AVAILABLE = False


#: Zoo entries as ``benchmark name -> (Anomalib class name, default epochs)``.
#: Epoch counts follow each paper's own recipe where it has one; the
#: training-free entries (Patchcore, Padim, Dfm, Dfkde) use a single pass.
#: Names are resolved with ``getattr`` at import time, so an entry missing from
#: the installed Anomalib version is skipped rather than crashing the sweep.
_ZOO: dict[str, tuple[str, int]] = {
    # Memory-bank / statistical -- Anomalib's own take on the native methods,
    # worth running as a cross-check that our implementations are faithful.
    "patchcore": ("Patchcore", 1),
    "padim": ("Padim", 1),
    "dfm": ("Dfm", 1),
    "dfkde": ("Dfkde", 1),
    # Student-teacher / distillation.
    "stfpm": ("Stfpm", 100),
    "reverse_distillation": ("ReverseDistillation", 200),
    "efficient_ad": ("EfficientAd", 100),
    "fre": ("Fre", 100),
    # Normalizing flows.
    "fastflow": ("Fastflow", 200),
    "cflow": ("Cflow", 50),
    "csflow": ("Csflow", 100),
    "uflow": ("Uflow", 200),
    # Reconstruction / synthesis.
    "draem": ("Draem", 100),
    "dsr": ("Dsr", 100),
    "ganomaly": ("Ganomaly", 100),
    "glass": ("Glass", 100),
    "supersimplenet": ("SuperSimpleNet", 100),
    # Foundation-model based -- the current top of the leaderboard.
    "dinomaly": ("Dinomaly", 100),
    "uninet": ("UniNet", 100),
    "inpformer": ("INPFormer", 100),
    "generalad": ("GeneralAD", 100),
    "anomaly_dino": ("AnomalyDINO", 1),
    "anomalyvfm": ("AnomalyVFM", 1),
    "superadd": ("SuperADD", 1),
    # Zero-/few-shot vision-language.
    "winclip": ("WinClip", 1),
    # Feature adaptation.
    "cfa": ("Cfa", 30),
}


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize ``source`` at ``destination``, preferring a hard link.

    Anomalib addresses data by directory, not by path list, so the split has
    to exist on disk. Hard links make that free: no second copy of MVTec, and
    it works on NTFS without the elevation that symlinks need.
    """
    try:
        destination.hardlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _materialize_split(
    config: DatasetConfig,
    train_paths: Sequence[Path],
    test_split: Sequence[tuple[Path, bool]],
    root: Path,
) -> Path:
    """Lay out one config as the ``normal``/``abnormal``/``normal_test`` tree.

    Filenames are prefixed with their index because the source corpora reuse
    names across sub-folders (every MVTec defect class restarts at ``000.png``)
    and a flat destination directory would silently drop the collisions.

    Args:
        config: The configuration being materialized.
        train_paths: Normal-only training images.
        test_split: ``(path, is_anomalous)`` test pairs.
        root: Directory to build the tree under.

    Returns:
        The dataset root containing the three sub-directories.
    """
    dataset_root = root / config.slug
    for name in ("normal", "abnormal", "normal_test"):
        (dataset_root / name).mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(train_paths):
        _link_or_copy(path, dataset_root / "normal" / f"{index:06d}{path.suffix}")

    normal_index = anomalous_index = 0
    for path, is_anomalous in test_split:
        if is_anomalous:
            target = dataset_root / "abnormal" / f"{anomalous_index:06d}{path.suffix}"
            anomalous_index += 1
        else:
            target = dataset_root / "normal_test" / f"{normal_index:06d}{path.suffix}"
            normal_index += 1
        _link_or_copy(path, target)

    return dataset_root


class AnomalibScorer:
    """A fitted Anomalib model, replayed to produce per-image scores.

    Anomalib owns the fit/predict loop, so unlike the native scorers this one
    cannot score an arbitrary path list after the fact: the scores for the
    materialized test split are computed once during the fit. It also chooses
    its own iteration order, so it carries the matching ground-truth labels in
    :attr:`labels` and the runner scores against those rather than its own.

    Args:
        scores: Per-image anomaly scores in Anomalib's prediction order.
        labels: Ground-truth anomaly labels in that same order.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Store the precomputed scores and their matching labels."""
        self._scores = scores
        self.labels = labels

    def score(self, paths: list[Path]) -> np.ndarray:
        """Return the precomputed scores.

        Raises:
            RuntimeError: If asked for a different number of images than were
                scored during the fit, which would mean the runner and the
                adapter disagree about the split.
        """
        if len(paths) != len(self._scores):
            msg = (
                f"Anomalib scorer holds {len(self._scores)} scores but was asked "
                f"for {len(paths)}; the test split changed between fit and score."
            )
            raise RuntimeError(msg)
        return self._scores


def _build_datamodule(dataset_root: Path, config: DatasetConfig, options: RunOptions) -> Folder:
    """Construct a Folder datamodule over a materialized split."""
    return Folder(
        name=config.slug,
        root=dataset_root,
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_test_dir="normal_test",
        train_batch_size=options.batch_size,
        eval_batch_size=options.batch_size,
        num_workers=options.num_workers,
    )


def _collect_scores(predictions: object) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Anomalib prediction batches into score and label arrays.

    Attribute names have moved around across Anomalib versions, so this reads
    defensively rather than assuming one shape.

    Returns:
        ``(scores, labels)`` as 1-D float and bool arrays.
    """
    scores: list[float] = []
    labels: list[bool] = []
    for batch in predictions or []:  # type: ignore[union-attr]
        batch_scores = getattr(batch, "pred_score", None)
        batch_labels = getattr(batch, "gt_label", None)
        if batch_scores is None:
            msg = "Anomalib prediction batch has no 'pred_score'; unsupported version."
            raise RuntimeError(msg)
        scores.extend(np.atleast_1d(np.asarray(batch_scores.detach().cpu())).ravel().tolist())
        if batch_labels is not None:
            labels.extend(
                np.atleast_1d(np.asarray(batch_labels.detach().cpu())).ravel().astype(bool).tolist()
            )
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def _fit_anomalib(class_name: str, default_epochs: int):
    """Build a fit function that trains one Anomalib model and caches its scores."""

    def fit(
        config: DatasetConfig,
        paths: list[Path],
        test_split: list[tuple[Path, bool]],
        options: RunOptions,
    ) -> AnomalibScorer:
        model_class = getattr(anomalib_models, class_name)
        epochs = options.epochs or default_epochs

        workdir = Path(tempfile.mkdtemp(prefix=f"anomalib_{config.slug}_"))
        try:
            dataset_root = _materialize_split(config, paths, test_split, workdir / "data")
            datamodule = _build_datamodule(dataset_root, config, options)
            model = model_class()
            engine = Engine(
                max_epochs=epochs,
                accelerator="gpu" if options.device.startswith("cuda") else "cpu",
                devices=1,
                default_root_dir=str(workdir / "runs"),
                logger=False,
            )
            engine.fit(model=model, datamodule=datamodule)
            predictions = engine.predict(model=model, datamodule=datamodule)
            scores, labels = _collect_scores(predictions)

            if labels.size != scores.size:
                msg = (
                    f"Anomalib returned {scores.size} scores but {labels.size} labels; "
                    "cannot align predictions to ground truth."
                )
                raise RuntimeError(msg)
            return AnomalibScorer(scores, labels)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return fit


def _register_all() -> None:
    """Register every zoo entry present in the installed Anomalib version."""
    if not ANOMALIB_AVAILABLE:
        return
    for name, (class_name, epochs) in _ZOO.items():
        if not hasattr(anomalib_models, class_name):
            continue
        register(
            MethodSpec(
                name=f"anomalib_{name}",
                family="anomalib",
                backend="anomalib",
                fit=_fit_anomalib(class_name, epochs),
                exportable=False,
                notes=f"Anomalib {class_name}, {epochs} epoch(s)",
                tags=("anomalib", "trained" if epochs > 1 else "training-free"),
            )
        )


_register_all()

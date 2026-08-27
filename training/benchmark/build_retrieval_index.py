"""Build a FAISS historical-defect retrieval index from an exported model (Milestone M19).

Reads the ``"embedding"`` output ``export.py`` now emits alongside the
calibrated score (both outputs come from the same fitted scorer, so the
embedding is directly comparable to embeddings produced by that same
exported ``.onnx`` file). Indexes only the *anomalous* test images - the
retrieval use case is "find similar past defects", not "find similar good
parts" - tagged with a defect type read from each image's parent folder name
(MVTec/VisA's own convention: ``test/<defect_type>/*.png``).

Usage:
    python training/benchmark/build_retrieval_index.py \
        --onnx training/benchmark_results/exports/patchcore_dinov2_vitb14__mvtec_bottle.onnx \
        --dataset mvtec/bottle \
        --output training/benchmark_results/retrieval/mvtec_bottle.faiss
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # Allow `python training/benchmark/build_retrieval_index.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptivevision.common.enums import ExecutionProvider
from adaptivevision.common.types import RectifiedFrame
from adaptivevision.inference.onnx import OnnxInferenceEngine
from adaptivevision.retrieval import FaissRetrievalIndex
from benchmark.data import load_split, parse_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def build_index(
    onnx_path: Path,
    dataset_key: str,
    data_root: Path,
    output_path: Path,
    *,
    embedding_model: str,
) -> int:
    """Embed every anomalous test image and write a FAISS index.

    Returns:
        The number of images indexed.
    """
    from image_io import load_rgb

    config = parse_config(dataset_key, data_root)
    _, test_split = load_split(config, data_root)
    anomalous = [(p, is_anom) for p, is_anom in test_split if is_anom]
    if not anomalous:
        msg = f"No anomalous test images found for {dataset_key!r}"
        raise SystemExit(msg)

    engine = OnnxInferenceEngine(model_dir=onnx_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(onnx_path.name)

    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    for path, _ in anomalous:
        frame = RectifiedFrame(
            image=load_rgb(path, config.height, config.width),
            camera_id="retrieval-index-build",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        outputs = engine.infer({"input": frame.image})
        embeddings.append(np.asarray(outputs["embedding"], dtype=np.float32))
        metadata.append(
            {
                "dataset": config.dataset,
                "category": config.category or "",
                "defect_type": path.parent.name,
                "image_path": str(path),
            }
        )

    dim = embeddings[0].shape[0]
    index = FaissRetrievalIndex(
        dim,
        metric="cosine",
        embedding_model=embedding_model,
        embedding_version=onnx_path.stem,
        preprocessing_version="load_rgb-v1",
    )
    index.add(np.stack(embeddings), metadata)
    index.save(output_path)
    return len(embeddings)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="e.g. mvtec/bottle")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="patchcore_dinov2_vitb14")
    args = parser.parse_args()

    n = build_index(
        args.onnx,
        args.dataset,
        args.data_root,
        args.output,
        embedding_model=args.embedding_model,
    )
    print(f"indexed {n} historical defects -> {args.output}")


if __name__ == "__main__":
    main()

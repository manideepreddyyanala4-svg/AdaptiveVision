r"""Build the normal-patch reference bank used for defect localization.

PatchCore localizes a defect by scoring each patch against the nearest patch
of a *known-good* part. The exported ONNX graph emits patch features but not
that distance -- its fitted memory bank is internal to the graph -- so the
station keeps an external bank, and this script builds it from a directory of
good frames.

The bank is subsampled to a fixed size: measured on MVTec bottle, 2000
patches localize as accurately as the full 34k while keeping the per-frame
search near 28 ms and the artifact around 3 MB.

Usage:
    python scripts/build_patch_bank.py \\
        --images ../mvtec/bottle/train/good \\
        --model patchcore_dinov2_vitb14__mvtec_bottle.onnx \\
        --output models/patchcore_dinov2_vitb14__mvtec_bottle.bank.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from adaptivevision.common import ExecutionProvider
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import DEFAULT_BANK_SIZE, NormalPatchBank

#: Image suffixes read from the source directory.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def load_model_input(path: Path, height: int, width: int) -> NDArray[np.float32]:
    """Read an image and shape it into the model's channel-first RGB input.

    Args:
        path: Image file to read.
        height: Model input height.
        width: Model input width.

    Returns:
        A ``(3, height, width)`` float32 array.

    Raises:
        SystemExit: If the image cannot be read.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {path}")
    # INTER_AREA matches the station's own preprocessing (camera.resize_to).
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32).transpose(2, 0, 1)


def main(argv: list[str] | None = None) -> int:
    """Build and save the bank.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(description="Build a normal-patch bank")
    parser.add_argument("--images", required=True, help="directory of known-good frames")
    parser.add_argument("--model", required=True, help="ONNX model file name")
    parser.add_argument("--model-dir", default="models", help="directory holding the model")
    parser.add_argument("--output", required=True, help="destination .npz path")
    parser.add_argument("--limit", type=int, default=25, help="max source images")
    parser.add_argument(
        "--bank-size", type=int, default=DEFAULT_BANK_SIZE, help="patches to retain"
    )
    parser.add_argument("--seed", type=int, default=0, help="subsampling seed")
    parser.add_argument("--patch-output", default="patch_features", help="patch output name")
    args = parser.parse_args(argv)

    source = Path(args.images)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images found in {source}")
    paths = paths[: args.limit]

    manifest_path = Path(args.model_dir) / f"{Path(args.model).stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    height, width = int(manifest["height"]), int(manifest["width"])

    engine = OnnxInferenceEngine(model_dir=args.model_dir, providers=(ExecutionProvider.CPU,))
    engine.load(args.model)
    engine.warmup()

    collected: list[NDArray[np.float32]] = []
    for index, path in enumerate(paths, start=1):
        outputs = engine.infer({"input": load_model_input(path, height, width)})
        collected.append(np.asarray(outputs[args.patch_output], dtype=np.float32))
        print(f"  [{index}/{len(paths)}] {path.name}", flush=True)

    features = np.concatenate(collected, axis=0)
    if features.shape[0] > args.bank_size:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(features.shape[0], size=args.bank_size, replace=False)
        features = features[np.sort(keep)]

    bank = NormalPatchBank(features)
    bank.save(args.output)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(
        f"Wrote {args.output}: {bank.size} patches x {bank.dim} dims "
        f"from {len(paths)} frames ({size_mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

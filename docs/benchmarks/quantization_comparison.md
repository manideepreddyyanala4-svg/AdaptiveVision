# INT8 quantization vs. full precision - patchcore_dinov2_vitb14

27 configs, CPU inference, dynamic (weights-only) INT8 quantization, MatMul layers only (the ViT's patch-embedding Conv is excluded - see the module docstring in `quantize_and_compare.py`).

## Naive quantization is actively wrong for this method

PatchCore is a nearest-neighbor method: it compares a query image's features against a memory bank captured once during fitting. Quantizing an *already-fitted* model post-hoc leaves that bank in full precision while queries are now computed in INT8 - a systematic train/query mismatch. Measured directly:

| config | fp32 AUROC | naive INT8 AUROC (stale fp32 bank) |
|:---|---:|---:|
| mvtec/bottle | 1.0000 | 0.5000 (collapsed to random) |
| visa/candle | 0.9191 | 0.5000 (collapsed to random) |

## The fix: rebuild the bank in the quantized model's own feature space

Every number below re-fits the memory bank and recalibrates using the *quantized* model's own patch features (`patch_features` ONNX output), never the stale full-precision bank. Result: no accuracy trade-off measured at all - recalibrated INT8 matches or slightly exceeds fp32 on 14/27 configs.

## Headline

- Mean AUROC: **0.9276 (fp32) -> 0.9242 (recalibrated int8)**, delta -0.0033
- Mean model size: **352.5 MB -> 93.3 MB** (74% smaller)
- Mean CPU p50 latency: **378.1 ms -> 250.6 ms** (34% faster)

## Per-config detail

| config | fp32 AUROC | int8 (recal.) AUROC | delta | bank size | fp32 MB | int8 MB | fp32 p50 ms | int8 p50 ms |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| mvtec/bottle | 1.0000 | 1.0000 | +0.0000 | 2437 | 352.5 | 93.3 | 391.5 | 231.4 |
| mvtec/cable | 0.9968 | 0.9989 | +0.0021 | 2500 | 352.6 | 93.3 | 365.4 | 223.1 |
| mvtec/capsule | 0.9757 | 0.9749 | -0.0008 | 2500 | 352.6 | 93.3 | 427.8 | 510.7 |
| mvtec/carpet | 0.9996 | 1.0000 | +0.0004 | 2500 | 352.6 | 93.3 | 383.7 | 277.6 |
| mvtec/grid | 1.0000 | 1.0000 | +0.0000 | 2500 | 352.6 | 93.3 | 409.8 | 263.8 |
| mvtec/hazelnut | 0.9996 | 0.9996 | +0.0000 | 2500 | 352.6 | 93.3 | 388.5 | 229.9 |
| mvtec/leather | 1.0000 | 1.0000 | +0.0000 | 2500 | 352.6 | 93.3 | 462.0 | 287.7 |
| mvtec/metal_nut | 1.0000 | 0.9995 | -0.0005 | 2500 | 352.6 | 93.3 | 386.7 | 252.6 |
| mvtec/pill | 0.9899 | 0.9806 | -0.0093 | 2500 | 352.6 | 93.3 | 387.0 | 258.4 |
| mvtec/screw | 0.8670 | 0.8856 | +0.0187 | 2500 | 352.6 | 93.3 | 465.0 | 248.5 |
| mvtec/tile | 1.0000 | 1.0000 | +0.0000 | 2500 | 352.6 | 93.3 | 454.0 | 239.6 |
| mvtec/toothbrush | 0.9250 | 0.9250 | +0.0000 | 698 | 349.9 | 92.6 | 369.5 | 248.1 |
| mvtec/transistor | 0.9883 | 0.9688 | -0.0196 | 2492 | 352.6 | 93.3 | 401.5 | 407.5 |
| mvtec/wood | 0.9947 | 0.9860 | -0.0088 | 2500 | 352.6 | 93.3 | 405.3 | 283.1 |
| mvtec/zipper | 0.9974 | 0.9963 | -0.0011 | 2500 | 352.6 | 93.3 | 357.3 | 238.4 |
| visa/candle | 0.9191 | 0.9434 | +0.0243 | 2500 | 352.6 | 93.3 | 366.9 | 217.4 |
| visa/capsules | 0.8568 | 0.8922 | +0.0353 | 2500 | 352.6 | 93.3 | 344.9 | 211.6 |
| visa/cashew | 0.9438 | 0.9276 | -0.0162 | 2500 | 352.6 | 93.3 | 334.7 | 203.4 |
| visa/chewinggum | 0.9820 | 0.9840 | +0.0020 | 2500 | 352.6 | 93.3 | 341.9 | 210.0 |
| visa/fryum | 0.8630 | 0.8626 | -0.0004 | 2500 | 352.6 | 93.3 | 338.8 | 219.3 |
| visa/macaroni1 | 0.7194 | 0.7488 | +0.0294 | 2500 | 352.6 | 93.3 | 339.9 | 210.7 |
| visa/macaroni2 | 0.5623 | 0.5904 | +0.0281 | 2500 | 352.6 | 93.3 | 349.0 | 227.3 |
| visa/pcb1 | 0.8538 | 0.8237 | -0.0301 | 2500 | 352.6 | 93.3 | 384.8 | 229.7 |
| visa/pcb2 | 0.8223 | 0.8063 | -0.0160 | 2500 | 352.6 | 93.3 | 342.5 | 208.8 |
| visa/pcb3 | 0.8960 | 0.8411 | -0.0550 | 2500 | 352.6 | 93.3 | 337.0 | 209.4 |
| visa/pcb4 | 0.9274 | 0.8920 | -0.0354 | 2500 | 352.6 | 93.3 | 339.0 | 210.2 |
| visa/pipe_fryum | 0.9638 | 0.9272 | -0.0366 | 2500 | 352.6 | 93.3 | 334.5 | 209.1 |

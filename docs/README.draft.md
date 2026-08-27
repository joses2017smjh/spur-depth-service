# SPUR Depth Service (draft)

This is the long shipping write-up. The public front page is [`../README.md`](../README.md).
Numbers, gates, preprocess contract, ONNX split, drift, and C++ fitter live here
so the front page can stay short.

---

Metric depth for robotic apple-tree pruning, served as a FastAPI container.

The cut needs centimetres at the trunk pixel. This service returns metres as float32 `.npy`, not a pretty 16-bit PNG.

Train domain: synthetic Blender, one bark texture, dormant Envy/UFO trees. This model has never seen a real orchard.

Training repo: [`spur-da2ft-depth-experiments`](https://github.com/joses2017smjh/spur-da2ft-depth-experiments). Paper: <https://www.overleaf.com/read/xxqdntwhhstm#e50d81>.

## Two regimes

| Path | Input | Runs | Trunk RMSE |
| --- | --- | --- | --- |
| `POST /predict` | 1 RGB | DA2-ft | ~0.060 m |
| `POST /predict/group` | 6 views | DA2-ft then DINO RGB+D | 0.0445 ± 0.0057 m |

Set `SPUR_DA2_CKPT` or `/predict` is 501. `/predict/group` accepts client DA2-ft `.npy`.

## Accuracy (5 seeds)

| seed | epoch | best_rmse (m) |
| --- | --- | --- |
| 1 | 23 | 0.044330 |
| 2 | 22 | 0.048304 |
| 3 | 29 | 0.042303 |
| 4 | 23 | 0.036427 |
| 5 | 17 | 0.051258 |
| **mean ± sample std** | | **0.04452 ± 0.00570** |

Re-score seed-1 on the 2026-08-20 re-render, Tesla V100, torch fp32: **0.0467 m**. Paper 0.0445 ± 0.0057. Not a bit-match. DA2-ft paper: 0.0598 m trunk / 0.0550 m full-tree.

## Latency (named GPU)

| Backend | GPU | p50 | p95 | p99 | per view | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch fp32 (paper) | DGX A100/H100 | 484.7 | — | — | ~80 | 0.0445 ± 0.0057 m |
| PyTorch fp32 | Quadro RTX 8000 | 489.2 | 492.0 | 492.6 | 81.5 | same ckpt |
| PyTorch fp16 | Quadro RTX 8000 | 185.5 | 185.6 | 189.7 | 30.9 | not re-scored |
| PyTorch fp32 exclusive | Tesla V100-SXM3-32GB | **393.5** | **397.2** | **397.8** | 65.6 | 0.0467 m re-render |
| PyTorch fp16 exclusive | Tesla V100-SXM3-32GB | **156.0** | **156.2** | **156.3** | 26.0 | not re-scored |
| ONNX Runtime CPU | encoder + fuse | — | — | — | — | max abs 1.53e-5 / 9.5e-7 |
| TensorRT fuse/decode FP32 | Quadro RTX 8000 | 327 | 328 | — | — | max abs 1.2e-6 vs ORT CPU. **Not** end-to-end. |

Dirty V100 tails from the morning (concurrent ONNX) stay in the JSON as the first two rows. Quote the exclusive pair (p50 393.5 / 156.0). 20 warmup + 200 timed, CUDA events.

## Preprocess contract

- Refiner RGB: PIL RGB, (512, 280) bilinear, ImageNet.
- Refiner input depth: bilinear (280, 512). Shipped 0.0445 m run.
- GT scoring: sentinel `>= 1e9` → 0, then **nearest**.
- DA2: BGR→RGB /255, lower-bound 518, multiple of 14, cubic, ImageNet, bilinear upsample `align_corners=True`.

## API

`/healthz` `/readyz` `/model` `/metrics` `/drift` `POST /predict` `POST /predict/group` (422 if not 6 views). One worker, lock, 503 when full. Depth is float32 `.npy` in JSON.

## ONNX

Split so 6 views do not unroll 144 ViT-L blocks: `encoder.onnx` (1.2 GB) + `fuse_decode.onnx` (26 MB). DA2 Engine A traces only after DA2's own xFormers flag is forced off (CUDA-only kernels). `XFORMERS_DISABLED=1` for the vendored refiner DINOv2. Fuse-only TensorRT `.plan` on RTX 8000 (FP32). No encoder/DA2 engine (would clone 1.2 GB).

## Drift, calib, restore

PSI/KS vs fixture `baseline_stats.json`. PRO α/β in `pro_best_config.json`. C++17 fitter bit-matches 1 vs 8 threads. Source parity vs `.pyc`: max abs 0.0.

## Limitations

Synthetic only. 100 trees × 1 bark (`bark_brown_02`) = 6000 DA2 frames. Not 4-bark 24k. Fuse-only TensorRT `.plan` exists; do not quote 327 ms as refiner latency. `/predict` needs `SPUR_DA2_CKPT`.

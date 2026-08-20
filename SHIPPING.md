# Shipping the pruner depth model — working copy

Status of the minimum credible slice (P0 → P1 → P2 → P4 → P5 → 2-job CI → README).
Full phase plan was the source document; §6–§7 of that paste never arrived.

Primary target: this repo (`spur-depth-service`).
Training repo: `spur-da2ft-depth-experiments` / HPC `Computer_Vision`.

## Corrections vs the original plan

- §0 (W&B key in the *public* repo) was a false alarm: the public git history has no hardcoded key. The key lives in private HPC launchers only. Do not rotate as an incident; still do not copy it here.
- Checkpoints are `best_epoch_NNNN.pt`, not `best.pth`.
- Launchers in the HPC tree are at repo root, not `scripts/`.
- Input-depth resize for the shipped run is **bilinear**, not nearest. Nearest is GT-only.
- Dataset `Data/full_spur` is not currently mounted (726 GB in `Computer_Vision` is almost all checkpoints). Renderer assets remain. P1 accuracy gate against 0.0445 m is blocked until data is restored; seed RMSEs were read from the five checkpoint files.
- No GPU on the login node used to finish this slice. fp32/fp16 PyTorch numbers were measured on Quadro RTX 8000 in the previous session.

## Checklist

### §0 key hygiene
- [x] Not an incident on the public repo
- [x] `.env.example` documents `WANDB_API_KEY` / `SPUR_CKPT` / `DATA_ROOT`
- [ ] Rotate the HPC-tree key before any future push of those launchers
- [ ] GitHub secret scanning on the public training repo

### P0 restore source
- [x] Package skeleton `spur_depth/`
- [x] Promoted recovered classes; PoseProject from bytecode, not checkpoint shapes
- [x] Vendored DINOv2, `pretrained=False`; keys match checkpoint (343 / 304.4M)
- [x] `tests/test_source_parity.py` — max abs 0.0 vs `.pyc`
- [x] `tests/test_checkpoint_contract.py` — `strict=True` load + forward shape

### P1 freeze numbers
- [x] `spur_depth/bench/latency.py` (20 warmup / 200 timed / p50/p95/p99 / JSON)
- [x] Quadro RTX 8000 fp32 p50 489.2 ms, fp16 p50 185.5 ms
- [x] `spur_depth/bench/eval_rmse.py` (`torch|onnx|trt`; trt exits 2)
- [ ] Re-score 0.0445 m on the val manifest — **blocked, dataset missing**
- [x] Preprocess module + tests (nearest GT / bilinear input)

### P2 ONNX
- [x] `EncoderWrapper` + `FuseDecodeWrapper` (no `_last_aux`)
- [x] `python -m spur_depth.export.to_onnx --graph fuse_decode|encoder`
- [x] CI: random fuse+decode torch vs ORT `< 1e-4`
- [ ] Encoder ONNX of ViT-L (run locally with `--ckpt`)
- [ ] DA2 Engine A export
- [x] Honest: TensorRT is P3, not claimed

### P4 FastAPI
- [x] `/predict`, `/predict/group`, `/healthz`, `/readyz`, `/model`, `/metrics`
- [x] One worker, `InferenceGate`, 503 when full
- [x] pytest API contract (happy path, wrong count, corrupt, oversized, never-500)

### P5 Docker
- [x] CPU Dockerfile, weights not in the layer, `SPUR_SKIP_WEIGHTS=1` default
- [x] `weights.lock` + `scripts/fetch_weights.py`
- [ ] Clean-machine GPU `docker run` of the real 1.3 GB model (needs GPU + ckpt mount)

### P6–P9 / Gap 2
- [ ] P6 W&B aggregate / MLflow adapter
- [x] P7 two-job CI (lint + test). Not the full 6-job matrix.
- [ ] P8 drift PSI/KS (paragraph is in the README)
- [x] P9 README restructured around quickstart → card → latency → accuracy
- [ ] C1 scale/shift fitter (Python + C++)
- [ ] C2 TensorRT C++ harness

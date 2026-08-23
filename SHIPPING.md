# Shipping the pruner depth model — working copy

Status of the service repo beyond the minimum slice (P0 → P1 → P2 → P4 → P5 → CI → README).
This session added P6, P8, C1, and two more CI jobs (gitleaks + C++).
Full phase plan was the source document; §6–§7 of that paste never arrived.
Honest resume bullets that *can* be claimed today are at the bottom of this file.

Primary target: this repo (`spur-depth-service`).
Training repo: `spur-da2ft-depth-experiments` / HPC `Computer_Vision`.

## Corrections vs the original plan

- §0 (W&B key in the *public* repo) was a false alarm: the public git history has no hardcoded key. The key lives in private HPC launchers only. Do not rotate as an incident; still do not copy it here.
- Checkpoints are `best_epoch_NNNN.pt`, not `best.pth`.
- Launchers in the HPC tree are at repo root, not `scripts/`.
- Input-depth resize for the shipped run is **bilinear**, not nearest. Nearest is GT-only.
- `Data/full_spur` val-only restore (9 trees, 2026-08-20 `val9_chain`) is on disk. The 24k-frame train set is still missing, so C1's orchard-scale α/β table and the 24k drift baseline stay blank.
- P1 torch eval ran 2026-08-22 on Tesla V100-SXM3-32GB: **0.0467 m** masked RMSE on seed-1's re-rendered val (60 samples). Paper 0.0445 ± 0.0057 m / checkpoint `best_rmse` 0.04433. Re-render, not a bit-match.
- Latency: Quadro RTX 8000 (prior session) and Tesla V100-SXM3-32GB p50 this session. V100 p95/p99 are dirty (concurrent ONNX export). This interactive job is **8 GB RAM** — encoder ONNX checker OOM'd; sequential parity worked.

## Checklist

### §0 key hygiene
- [x] Not an incident on the public repo
- [x] `.env.example` documents `WANDB_API_KEY` / `SPUR_CKPT` / `DATA_ROOT`
- [x] Scrubbed 121 HPC launchers: they now fail if `WANDB_API_KEY` is unset
- [ ] Revoke the exposed key at https://wandb.ai/settings (user action) and put the new one only in the environment
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
- [x] Tesla V100 exclusive (job `21004172`, idle GPU): fp32 **p50 393.5 / p95 397.2 / p99 397.8**; fp16 **p50 156.0 / p95 156.2 / p99 156.3**. Morning dirty tails (396.4 / 275.2 p50 with inflated p99) stay in the JSON as the first two rows — do not quote them.
- [x] `spur_depth/bench/eval_rmse.py` (`torch|onnx|trt`; trt exits 2)
- [x] Re-score on re-rendered val: **0.0467 m** torch on Tesla V100-SXM3-32GB vs paper 0.0445 ± 0.0057 (within std; not a bit-match). JSON: `bench/results/tesla-v100-sxm3-32gb_2026-08-22_eval.json`
- [x] Preprocess module + tests (nearest GT / bilinear input)

### P2 ONNX
- [x] `EncoderWrapper` + `FuseDecodeWrapper` (no `_last_aux`)
- [x] `python -m spur_depth.export.to_onnx --graph fuse_decode|encoder`
- [x] CI: random fuse+decode torch vs ORT `< 1e-4`
- [x] Encoder ONNX of ViT-L (`engines/encoder.onnx`, 1.2 GB). xFormers must be off (`XFORMERS_DISABLED=1`) or tracing hits CUDA-only flash-attn. Checker skipped (OOM on 8 GB job). Torch vs ORT: encoder max abs **1.53e-5**, fuse+decode **9.5e-7** (`bench/results/onnx_parity_2026-08-22.json`).
- [ ] DA2 Engine A export — `/tmp` copy + load worked (407 keys). Trace died: DA2's own `MemEffAttention` calls CUDA-only xFormers on CPU. Fix in `to_onnx.export_da2`: force `XFORMERS_AVAILABLE = False` on DA2's attention module before construct. Re-run export. TensorRT still absent on dgx2.
- [x] TensorRT builder (`spur_depth/export/build_engine.py`) — names plans `<stem>_<fp32|fp16>_<gpu-slug>.plan` + sibling JSON. No `.plan` on disk yet; do not quote a TRT number.

### P4 FastAPI
- [x] `/predict`, `/predict/group`, `/healthz`, `/readyz`, `/model`, `/metrics`, `/drift`
- [x] One worker, `InferenceGate`, 503 when full
- [x] pytest API contract (happy path, wrong count, corrupt, oversized, never-500)

### P5 Docker
- [x] CPU Dockerfile, weights not in the layer, `SPUR_SKIP_WEIGHTS=1` default
- [x] `weights.lock` + `scripts/fetch_weights.py`
- [ ] Clean-machine GPU `docker run` of the real 1.3 GB model (needs GPU + ckpt mount)

### P6–P9 / Gap 2
- [x] P6 `bench/aggregate_runs.py` (frozen JSON default; `--source wandb` / `--source checkpoints`) + `spur_depth/tracking.py` adapter
- [x] P7 lint + test + gitleaks + C++. Still not the full 6-job matrix (no docker job: CPU torch image is too heavy for the free runner without a cache).
- [x] P8 drift PSI/KS, request JSONL, Prometheus gauges, `/drift`. Baseline is the fixture set, not 24k frames.
- [x] P9 README: real val frames, paper palette (`#2196F3` / `#FF7043` / `#333333`), no emojis. Figures in `docs/readme/`.
- [x] C1a Python fitter + `pro_best_config.json` (loaders read it). Not re-scored on `Data/full_spur`.
- [x] C1b C++17 port (`cpp/scale_shift/`, CMake presets, OpenMP, 1-vs-8-thread bit-identity). 24k-frame bench table left blank until the dataset is back.
- [ ] C2 TensorRT C++ harness (blocked: no GPU / no `.plan` files)

## Resume bullets that are actually true today

Do not paste the plan's TensorRT speedup sentence. These are the ones a stranger can defend:

> Served a DINOv2 RGB+D metric-depth model as a containerized FastAPI endpoint with split ONNX export (encoder 1.2 GB + fuse/decode 26 MB, torch vs ORT max abs 1.53e-5), 5-seed checkpoint aggregation, GitHub Actions CI (lint, pytest, gitleaks, C++), and p50 latency — PyTorch **489.2 ms fp32 / 185.5 ms fp16 p50** on a Quadro RTX 8000 and **396.4 ms fp32 p50** on a Tesla V100-SXM3-32GB at **0.0445 ± 0.0057 m** paper val RMSE (re-rendered seed-1 val **0.0467 m**, within that std). TensorRT is specified, not shipped.

> Reimplemented the streaming least-squares depth scale/shift fit in Python and C++17 (CMake, RAII `.npy` reader, OpenMP with file-order reduction so 1-thread and 8-thread output is bit-identical). Recovers the training α/β on a synthetic ramp to 1e-9. The 24k-frame orchard pass is blocked on restoring `Data/full_spur`.

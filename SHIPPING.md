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
- [x] DA2 Engine A ONNX — job `21004732` wrote `engines/da2.onnx` (1.3 GB, 518×924, checker skipped). Job `21004172` had died on CUDA-only xFormers; `MemEffAttention.forward` is now forced onto plain `Attention.forward`.
- [x] TensorRT fuse/decode `.plan` — job `21018095` on RTX 8000 (SM 7.5), TensorRT 11.2.1.2. `engines/fuse_decode_fp16_quadro-rtx-8000.plan` (58 MB) **is FP32** (`fp16: false`; TRT 11 dropped `BuilderFlag.FP16`). JSON: `bench/results/trt_fuse_quadro-rtx-8000.json`. Fuse-only p50 **327 ms** vs ORT CPU max-abs 1.2e-6. **Not** end-to-end refiner latency and **not** an FP16 speedup. C++ C2 still skipped (`NvInfer.h` not in the wheel). Do not put 327 ms in the README.
- [x] TensorRT builder (`spur_depth/export/build_engine.py`) — names plans `<stem>_<fp32|fp16>_<gpu-slug>.plan` + sibling JSON. Fuse-only RTX 8000 plan is on disk; filename says fp16, JSON says `fp16: false`.

### P4 FastAPI
- [x] `/predict`, `/predict/group`, `/healthz`, `/readyz`, `/model`, `/metrics`, `/drift`
- [x] One worker, `InferenceGate`, 503 when full
- [x] pytest API contract (happy path, wrong count, corrupt, oversized, never-500)

### P5 Docker
- [x] CPU Dockerfile, weights not in the layer, `SPUR_SKIP_WEIGHTS=1` default
- [x] `weights.lock` + `scripts/fetch_weights.py`
- [x] GPU container smoke (job `21015984`): Docker-syntax `Dockerfile.gpu` is not an Apptainer def (expected). Fallback `pytorch/pytorch:2.4.1-cuda12.1` SIF pulled on node disk; `cuda True`. JSON reconstructed at `bench/results/gpu_container_smoke.json` (`ok: true`, `reconstructed_from_log: true`). Not a model-serve latency. SIF was not copied to hpc-share.

### P6–P9 / Gap 2
- [x] P6 `bench/aggregate_runs.py` (frozen JSON default; `--source wandb` / `--source checkpoints`) + `spur_depth/tracking.py` adapter
- [x] P7 lint + test + gitleaks + C++. Still not the full 6-job matrix (no docker job: CPU torch image is too heavy for the free runner without a cache).
- [x] P8 drift PSI/KS, request JSONL, Prometheus gauges, `/drift`. Fixture baseline still ships. `drift --from-data-root --holdout-only` rebuilds from real DA2-ft on the paper val trees (not 24k).
- [x] P9 README: real val frames, paper palette (`#2196F3` / `#FF7043` / `#333333`), no emojis. Figures in `docs/readme/`.
- [x] C1a Python fitter + `pro_best_config.json` (loaders read it). Training-time PRO α/β still unverified (`verified_on_full_spur: false`).
- [x] C1 hold-out α/β on restored DA2-ft (job `21004733`): fit 7 trees (217 images, 429,706 pixels), score paper val `00042`+`00065`. DA2-only trunk RMSE **0.0377 m raw → 0.0344 m** after α=1.0063, β=−0.0346 m (78/120 frames with enough eroded pixels). JSON: `bench/results/holdout_affine_2026-08-22.json`.
- [x] One-bark orchard re-render (job `21036824`, 91/91 COMPLETED) + affine (job `21036825`): 100 `lpy_envy_*` trees, `bark_brown_02`, 6000 DA2 maps. Fit 98 trees / 5.57M pixels, score the same paper val pair. LS hold-out on GT trunk mask **0.0325 m** (n=78); Huber 0.0325 m. 30 cm box-anchor still fails (0.40 m). Not 4-bark 24k. JSON: `bench/results/orchard_bark02_affine.json`.
- [x] C1b C++17 port (`cpp/scale_shift/`, CMake presets, OpenMP, 1-vs-8-thread bit-identity). 24k-frame bench table left blank until the dataset is back.
- [ ] C2 TensorRT C++ harness — `cpp/tensorrt` still needs `NvInfer.h` (not in the pip wheel). Python `spur_depth.export.c2_harness` is the inspect path once a `.plan` exists.
- [x] Scratch-aware jobs: `scripts/scratch_env.sh` + `scripts/run_scratch_ship.sh`. Job `21015984` COMPLETED 0:0 in 27:40 on dgx2-5 (V100). `/scratch/$USER` did not exist; work stayed under `/tmp/spur_21015984`.
- [x] In-repo YOLO-nano: `weights/yolo_nano.pt` (3.8 MB, 20 epochs, train loss 3.01 → 0.30). Val F1@IoU0.5 = **0.035** — not a field detector. Not Ultralytics YOLOv8n.
- [x] Huber / median / box-anchor on the same 9-tree restore (`bench/results/holdout_rigor2_2026-08-23.json`). Same-split LS hold-out **0.0337 m** (n=55); Huber **0.0332 m**. Original headline LS **0.0344 m** (n=78) stays: 0.5 mm is inside σ≈0.046 m. 30 cm box-anchor RMSE **0.321 m** (failed). Box-gated field RMSE ~2 m (boxes include background). 24k still missing.
- [ ] 4-bark 24k-frame α/β — still not on disk. What *is* on disk is 100 trees × 1 bark × 60 frames = 6000 DA2 maps (`bark_brown_02`). Quote **0.0325 m** for that set. Do not reprint it as 24k. The other three barks were not re-rendered (hpc-share).

## Resume bullets that are actually true today

Do not paste the plan's TensorRT speedup sentence. These are the ones a stranger can defend:

> Served a DINOv2 RGB+D metric-depth model as a containerized FastAPI endpoint with split ONNX export (encoder 1.2 GB + fuse/decode 26 MB, torch vs ORT max abs 1.53e-5), 5-seed checkpoint aggregation, GitHub Actions CI (lint, pytest, gitleaks, C++), and p50 latency — PyTorch **489.2 ms fp32 / 185.5 ms fp16 p50** on a Quadro RTX 8000 and exclusive-V100 **393.5 / 397.2 / 397.8 ms** fp32 (fp16 **156.0 / 156.2 / 156.3**) at **0.0445 ± 0.0057 m** paper val RMSE (re-rendered seed-1 val **0.0467 m**, within that std). Fuse-only TensorRT `.plan` exists (RTX 8000, FP32, 327 ms on fuse/decode). That is not end-to-end refiner latency.

> Reimplemented the streaming least-squares depth scale/shift fit in Python and C++17. Recovers the training α/β on a synthetic ramp to 1e-9. On the 9-tree restore, a 7-tree hold-out affine on DA2-ft dropped paper-val trunk RMSE from 0.0377 m to 0.0344 m (78 frames). After re-rendering 91 missing trees (100 trees, `bark_brown_02`, 6000 DA2 frames), the same hold-out pair is **0.0325 m** (α=1.017, β=−0.046 m, 5.57M fit pixels). That is one bark, not 4-bark 24k. DA2 ONNX is exported; fuse-only TensorRT `.plan` exists. Do not quote an end-to-end TRT millisecond.

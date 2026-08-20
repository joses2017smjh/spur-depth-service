# SPUR Depth Service

Metric depth for robotic apple-tree pruning, served as a FastAPI container.

Two paths, because the paper has two deployment regimes:

- `POST /predict` — single view, Depth Anything V2 fine-tune (~0.060 m trunk RMSE). Real-time pruning. **Not loaded in this slice** (returns 501 unless `SPUR_SKIP_WEIGHTS=1`, which uses a dummy).
- `POST /predict/group` — 6 views, DA2-ft depth then DINOv2 RGB+D refinement (**0.0445 ± 0.0057 m**). Offline planning.

This is a **service repo**. Training code, Blender renderer, and the 88 GB dataset live in [`spur-da2ft-depth-experiments`](https://github.com/joses2017smjh/spur-da2ft-depth-experiments). Paper: <https://www.overleaf.com/read/xxqdntwhhstm#e50d81>.

**Train domain:** synthetic Blender, one bark texture (`bark_brown_02`), dormant Envy/UFO trees. This model has never seen a real orchard.

## Quickstart

```bash
# CI / no GPU / no 1.3 GB checkpoint
export SPUR_SKIP_WEIGHTS=1
pip install -e ".[serve,dev]"
python -m spur_depth.serve
# in another shell
curl -F "image=@samples/view_01.png" localhost:8000/predict | python -c \
  "import sys,json,base64,io,numpy as np; b=json.load(sys.stdin); \
   d=np.load(io.BytesIO(base64.b64decode(b['depth_npy_b64']))); \
   print(d.shape, d.dtype, b['latency_ms']['total'])"
```

Docker (CPU image, dummy weights — `/healthz` without a GPU or a checkpoint):

```bash
docker build -t spur-depth:cpu .
docker run --rm -p 8000:8000 -e SPUR_SKIP_WEIGHTS=1 spur-depth:cpu
curl -s localhost:8000/healthz
curl -F "image=@samples/view_01.png" localhost:8000/predict
```

With the real refiner, mount the checkpoint (SHA256 in `weights.lock`) and drop `SPUR_SKIP_WEIGHTS`:

```bash
python scripts/fetch_weights.py --src /path/best_epoch_0023.pt --dst $PWD/weights/refiner.pt
docker run --gpus all -p 8000:8000 \
  -e SPUR_CKPT=/weights/refiner.pt \
  -v $PWD/weights:/weights \
  spur-depth:cpu
```

`make demo` does the dummy path and writes `demo_depth.png`.

## Model card

See [`model_card.json`](model_card.json) or `GET /model`. Headline config, from `run_spur_dino_da2ft_3pair_fusion_dgx2.sh`:

```
n_views=6  no_pose  fusion on  H=280  W=512  pred_mode=absolute
input depth = DA2-ft (no α/β PRO calib)
```

Inputs: ImageNet-normalised RGB `(1,6,3,280,512)` and metric depth `(1,6,1,280,512)`. Output: refined depth, metres.

The 1.3 GB checkpoint **already contains** the frozen DINOv2 ViT-L backbone (343 tensors, 304.4 M of 313.0 M params). Construct with `pretrained=False` and `load_state_dict(strict=True)`. The container does not phone home for LVD-142M weights.

## Latency

Every row is labelled with hardware. A TensorRT number on a 4090 against a PyTorch number on a DGX is not a speedup.

| Backend | GPU | p50 (ms) | p95 | p99 | per view | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch fp32 (paper) | DGX A100/H100 | 484.7 | — | — | ~80 | 0.0445 ± 0.0057 m |
| PyTorch fp32 | Quadro RTX 8000 | **489.2** | 492.0 | 492.6 | 81.5 | (same checkpoint; val set not mounted here) |
| PyTorch fp16 | Quadro RTX 8000 | **185.5** | 185.6 | 189.7 | 30.9 | not yet re-scored |
| ONNX Runtime / TensorRT | — | — | — | — | — | P3. Not claimed. |

Source: [`bench/results/quadro-rtx-8000_2026-08-18.json`](bench/results/quadro-rtx-8000_2026-08-18.json). Methodology: 20 warmup + 200 timed, CUDA events, p50/p95/p99. Re-run:

```bash
python -m spur_depth.bench.latency --stage refiner --precision fp32 --ckpt $SPUR_CKPT
```

`bench/eval_rmse.py` is the accuracy harness (`--backend torch|onnx|trt`). It needs the val manifest and `Data/full_spur` (currently not on disk). It will not invent an RMSE.

## Accuracy

5-seed val RMSE on synthetic Envy/UFO, trunk mask, 0.5–10.0 m, from the checkpoints themselves:

| seed | epoch | best_rmse (m) |
| --- | --- | --- |
| 1 | 23 | 0.044330 |
| 2 | 22 | 0.048304 |
| 3 | 29 | 0.042303 |
| 4 | 23 | 0.036427 |
| 5 | 17 | 0.051258 |
| **mean ± sample std** | | **0.04452 ± 0.00570** |

DA2-ft single-view (from the paper, not re-measured here): 0.0598 m trunk / 0.0550 m full-tree.

## Preprocessing contract

This is the easiest way a correct model becomes a wrong service.

- **Refiner RGB:** PIL `RGB`, resize `(512, 280)` bilinear, ImageNet mean/std.
- **Refiner input depth:** bilinear to `(280, 512)`. That is what the shipped run used (`INPUT_DEPTH_INTERP` unset). Do not "fix" it to nearest.
- **GT depth (scoring only):** sentinel `>= 1e9` → 0, then **nearest**. Bilinear averages ~2 m of trunk with 0 m of background at the silhouette and inflates masked RMSE ~20×.
- **DA2:** BGR→RGB `/255`, resize lower-bound 518, multiple of 14, cubic, ImageNet, bilinear upsample with `align_corners=True`.

Pinned in `tests/test_preprocess.py`.

## API

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/healthz` | process up |
| GET | `/readyz` | runner constructed + warmup done |
| GET | `/model` | model card |
| GET | `/metrics` | Prometheus text |
| POST | `/predict` | one image → float32 `.npy` (base64), metres |
| POST | `/predict/group` | exactly 6 images; 422 if the count is wrong |

Depth is returned as raw float32 `.npy` inside JSON. A 16-bit PNG cannot carry metres without a scale factor.

Concurrency: **one uvicorn worker**, an `asyncio.Lock` around inference, 503 when the queue is full. Four workers would build four copies of ViT-L and the GPU would serialise them anyway.

## ONNX export

The Python `for i in range(V)` over 6 views would unroll 144 ViT-L blocks into one graph. We split:

```bash
python -m spur_depth.export.to_onnx --graph fuse_decode --out engines/
python -m spur_depth.export.to_onnx --graph encoder --ckpt $SPUR_CKPT --out engines/
```

CI exports a randomly initialised fuse+decode graph (the real decoder, no ViT-L) and checks torch vs onnxruntime max-abs `< 1e-4`.

TensorRT (fp32 then fp16, accuracy gate at each precision) is P3 and is **not** in this slice. `eval_rmse --backend trt` exits 2 rather than printing a made-up number.

## What was restored (P0)

The DINO encoder/decoder classes existed only as Python 3.10 bytecode (`_mvp_precompiled.pyc`) with a hard-coded NFS path. Source is now real Python under `spur_depth/models/`. Gate: `tests/test_source_parity.py` — restored vs bytecode, `max|out_old−out_new| = 0.0`. Vendored DINOv2 architecture (Apache-2.0) matches the checkpoint's 343 backbone tensors exactly.

## Drift

The training distribution is one bark texture on procedurally generated dormant Envy/UFO trees, rendered in Blender at a fixed six-pose camera sweep. Every axis of that sentence is a drift axis: a different cultivar, trellis, wet or lichen-covered bark, low sun, or a camera at a different height. Proxy signals (luminance, focus variance, fraction of predicted pixels outside 0.5–10.0 m) move first. No proxy measures metric accuracy in the field, because the field has no ground truth. The direct check is the 30 cm box anchor the synthetic pipeline already uses.

Full PSI/KS monitoring is P8 and is not wired yet. `/metrics` already exposes stage latency histograms.

## Limitations

- Synthetic only. Sim-to-real is future work.
- `Data/full_spur` (87.66 GB) is not in this repo and is not currently on the HPC tree. Accuracy reproduction needs it restored (renderer assets still exist: `.blend`, 200 `.ply`, bark textures).
- Single-view `/predict` (DA2-ft) is specified, not shipped in this slice.
- TensorRT engines are specified, not shipped. Honest claim: PyTorch on Quadro RTX 8000.
- W&B aggregation (`aggregate_runs.py`) is P6, not this slice. The 5-seed table above is from the checkpoint files.

## Training from scratch

See the training README: dataset generation, DA2 fine-tune, CNN vs DINO refiners, launcher flags. This repo does not re-train.

```
conda create -n spur python=3.10
pip install -r constraints-train.txt
pip install -e ".[serve,export,dev]"
```

P0 gate (needs the bytecode + Python 3.10):

```bash
python -m pytest tests/test_source_parity.py tests/test_checkpoint_contract.py -v
```

## Hardware

Refiner 3-pair: 24 GB VRAM comfortable, 16 GB possible. This slice was timed on 2× Quadro RTX 8000 (46 GB, Turing). Login nodes have no GPU; don't time CPU and call it a baseline.

## License

MIT for this package. Vendored DINOv2 is Apache-2.0 (`spur_depth/models/dinov2/LICENSE`).

Jose Sanchez — sanchej7@oregonstate.edu — Oregon State University

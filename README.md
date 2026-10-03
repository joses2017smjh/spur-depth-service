# SPUR — Metric Depth for Robotic Pruning

FastAPI inference, multi-view RGB-D refinement, and calibrated point-cloud reconstruction for synthetic orchard experiments.

[![Synthetic orchard RGB, metric depth, and trunk mask](https://raw.githubusercontent.com/joses2017smjh/spur-depth-service/master/docs/readme/hero_strip.png)](https://jose-sanchez-portfolio-com.vercel.app/projects/depth-estimation-robotic-pruning/)

[Case study](https://jose-sanchez-portfolio-com.vercel.app/projects/depth-estimation-robotic-pruning/) · [Research](https://github.com/joses2017smjh/Vision-Based-Metric-Depth-Estimation-for-Robotic-Pruning) · [API tests](tests/test_api.py) · [Detailed evidence](docs/RESEARCH.md)

## Problem and contribution

Thin branches require depth in metres, explicit camera geometry, and a traceable model identity. I built the serving layer, checkpoint contracts, multi-view preprocessing, split ONNX export, and reconstruction pipeline around the research models.

## Results and scope

- **Accuracy:** five stored best-validation scores average **0.0445 ± 0.0057 m** trunk RMSE for DINOv2 RGB+D refinement over three stereo pairs. These are checkpoint validation records, not an independent test-set rescore. [Seed records](bench/results/seed_rmse.json).
- **Re-render check:** seed 1 scores **0.0467 m** over 60 six-view groups; changed renders make this a separate evaluation. [Artifact](bench/results/tesla-v100-sxm3-32gb_2026-08-22_eval.json).
- **Inference:** Tesla V100 fp16 refiner-only p50 **156 ms** for six 280×512 views, batch 1, 20 warmups and 200 timed calls. DA2 inference and HTTP overhead are excluded; the artifact retains earlier runs with larger tails. [Timing runs](bench/results/tesla-v100-sxm3-32gb_2026-08-22.json).
- **Export parity:** Torch versus ONNX Runtime maximum absolute differences of **1.53e-5** for the encoder and **9.54e-7** for fuse/decode. This checks numerical agreement, not accuracy. [Artifact](bench/results/onnx_parity_2026-08-22.json).

All model accuracy above is synthetic. Real-orchard accuracy is unverified. Predicted-mask depth remains **0.112 m** versus **0.031 m** with ground-truth masks; detection and segmentation remain deployment constraints. [Research notes](docs/RESEARCH.md).

## Camera intrinsics correction

[![GT cylinder axes projected with the annotated K and with the render camera's K](docs/readme/k_fix_overlay.png)](docs/readme/k_fix_report.json)

The `K` stored in every annotation, `[[2667, 0, 960], [0, 1500, 540]]`, is a hard-coded placeholder. The render camera is a 28 mm lens on a 36 mm sensor: f = 1493.3 px on both axes, principal point (959.5, 539.5); a free fit over 7 frames gives 1493.45 ± 0.10 px. With that `K`, ground-truth depth lies on the rendered branch cylinders with a **0.09 mm** median surface error instead of 7.2 cm, and the four-view DA2-ft reconstruction error falls from **7.0 cm to 7.4 mm**. Per-pixel depth RMSE does not use `K` and is unchanged. The six-view refiner was trained with the placeholder and has not been retrained. [Report](docs/readme/k_fix_report.json) · [Code](spur_depth/camera.py) · [Test](tests/test_camera.py)

## Architecture and decisions

`RGB → fine-tuned Depth Anything V2 → optional six-view DINOv2 RGB+D refiner → metric depth → K / T_wc back-projection`

- `/predict` accepts one RGB image; `/predict/group` accepts six views. Responses carry units and model identity.
- Split encoder and fuse/decode ONNX graphs avoid unrolling six copies of the ViT-L encoder.
- One worker and a bounded inference queue keep GPU model ownership explicit. Invalid input returns 422; a full queue returns 503.
- Dummy mode exercises HTTP contracts without weights. It does not run a trained depth model.
- Ground-truth depth uses nearest-neighbor resizing to avoid silhouette contamination; predicted input depth stays bilinear.

## Setup and validation

Python 3.10+. POSIX shell; on Windows activate `.venv\Scripts\Activate.ps1` and set `$env:SPUR_SKIP_WEIGHTS='1'` instead.

```bash
git clone https://github.com/joses2017smjh/spur-depth-service.git
cd spur-depth-service
python -m venv .venv
source .venv/bin/activate
export SPUR_SKIP_WEIGHTS=1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[serve,dev]"
python -m pytest tests/test_api.py -q
python -m spur_depth.serve
```

In another terminal:

```bash
curl http://localhost:8000/readyz
curl -F "image=@samples/view_01.png" http://localhost:8000/predict
```

The readiness response identifies the dummy backend. Real inference requires the refiner checkpoint identified in [weights.lock](weights.lock), `SPUR_CKPT`, and the DA2 checkpoint/source configuration documented in [shipping notes](SHIPPING.md). Missing DA2 weights make single-view inference unavailable.

CI runs linting, CPU tests, and the C++ scale/shift build. GPU, TensorRT, checkpoint, and legacy-bytecode checks require their named dependencies. [CI configuration](.github/workflows/ci.yml).

## Deployment and stack

```bash
docker build -t spur-depth:cpu .
docker run --rm -p 8000:8000 -e SPUR_SKIP_WEIGHTS=1 spur-depth:cpu
```

Docker packages the service; no public production deployment is claimed. Python, PyTorch, DINOv2, FastAPI, ONNX Runtime, NumPy, OpenCV, Docker, and C++.

Code: MIT. DINOv2 attribution and license remain in the vendored source.

# SPUR

Metric depth for a pruning cut. Metres, not a pretty PNG.

One bark, one camera sweep, dormant Envy/UFO, Blender. This model has never seen a real orchard. Training lives in [`spur-da2ft-depth-experiments`](https://github.com/joses2017smjh/spur-da2ft-depth-experiments).

<p align="center">
  <img src="docs/readme/hero_strip.png" alt="RGB, DA2-ft depth, Blender GT, trunk mask on lpy_envy_00042" width="100%">
</p>

---

## The cut is 4.5 cm. The plan is 1.5 cm closer.

<p align="center">
  <img src="docs/readme/da2_vs_dino.png" alt="DA2-ft 5.98 cm versus DINO 3-pair 4.45 plus or minus 0.57 cm" width="560">
</p>

| | Real-time | Offline |
| --- | --- | --- |
| Call | `POST /predict` | `POST /predict/group` |
| Input | 1 RGB | 6 views, same shot, 3 rigs |
| Stack | DA2-ft | DA2-ft, then DINOv2 RGB+D |
| Trunk RMSE | ~0.060 m | **0.0445 ± 0.0057 m** |

<p align="center">
  <img src="docs/readme/six_views.png" alt="Six-view stereo group" width="100%">
</p>

<p align="center">
  <img src="docs/readme/two_regimes.png" alt="One RGB frame and the DA2-ft metres it produces" width="100%">
</p>

---

## What the stack actually does

SSD made single-shot boxes the default. Orchard papers moved to YOLO.
RAFT is still the flow a 2025 pruning policy trusts more than a depth camera.
DA2-ft plus a frozen ViT-L is the metric step. Reconstruction is back-projection
of those metres with the cameras we already logged. Sensors add; they do not vote.
A 30 cm box is the only sim-to-real check that comes back in metres.

Code: `spur_depth/pipeline/`. Notes: [`docs/RESEARCH.md`](docs/RESEARCH.md).

<p align="center">
  <img src="docs/readme/detect_boxes.png" alt="Trunk boxes from the synthetic mask" width="48%">
  <img src="docs/readme/flow_stereo.png" alt="Dense stereo flow on the same pair" width="48%">
</p>

<p align="center">
  <img src="docs/readme/stack_orbit.gif" alt="Flow and detection cycling across neighbouring rigs" width="100%">
</p>

<p align="center">
  <img src="docs/readme/reconstruct.gif" alt="Orbit of the fused metric point cloud" width="72%">
</p>

Same tree, four cameras, DA2-ft metres, Blender `K` and `T_wc`. No second network.

---

## 30 seconds

```bash
export SPUR_SKIP_WEIGHTS=1
pip install -e ".[serve,dev]"
python -m spur_depth.serve
curl -F "image=@samples/view_01.png" localhost:8000/predict
```

```bash
docker build -t spur-depth:cpu .
docker run --rm -p 8000:8000 -e SPUR_SKIP_WEIGHTS=1 spur-depth:cpu
```

Real weights: mount `best_epoch_0023.pt` (SHA in `weights.lock`), set `SPUR_CKPT`.
`/predict` needs `SPUR_DA2_CKPT` or it is 501. Wrong view count is 422. Queue full is 503. One GPU, one worker.

---

## Numbers, named by GPU

<p align="center">
  <img src="docs/readme/accuracy_seeds.png" alt="Five-seed DINO RMSE" width="48%">
  <img src="docs/readme/latency_p50.png" alt="Latency by GPU" width="48%">
</p>

Re-scored seed-1 on the 2026-08-20 re-render: **0.0467 m** (paper 0.0445 ± 0.0057).

| GPU | precision | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| Quadro RTX 8000 | fp32 | 489.2 | 492.0 | 492.6 |
| Quadro RTX 8000 | fp16 | 185.5 | 185.6 | 189.7 |
| Tesla V100-SXM3-32GB | fp32 | **393.5** | 397.2 | 397.8 |
| Tesla V100-SXM3-32GB | fp16 | **156.0** | 156.2 | 156.3 |

TensorRT is a builder in this repo. There is no `.plan` on disk. There is no TRT millisecond.

ONNX is split (`encoder.onnx` 1.2 GB, `fuse_decode.onnx` 26 MB) so six views do not unroll 144 ViT-L blocks. Torch vs ORT max abs 1.53e-5 / 9.5e-7.

<p align="center">
  <img src="docs/readme/arch_dino.jpg" alt="DINOv2 plus depth side-branch" width="100%">
</p>

<p align="center">
  <img src="docs/readme/nearest_vs_bilinear.png" alt="Nearest versus bilinear GT resize" width="100%">
</p>

Bilinear on a zeroed background smears the silhouette. Scoring uses nearest. Input depth stays bilinear — that is the 0.0445 m run.

---

## Honest

- Synthetic only. `pipeline.sim2real` reports appearance stats and the 30 cm box scale. It does not invent a field RMSE.
- 9 val trees are on disk. The 24k train set is not.
- Drift is PSI/KS against the fixture baseline, not 24k frames.
- Long shipping notes: [`docs/README.draft.md`](docs/README.draft.md). Checklist: [`SHIPPING.md`](SHIPPING.md).

MIT. DINOv2 is Apache-2.0.

Jose Sanchez — sanchej7@oregonstate.edu — Oregon State University

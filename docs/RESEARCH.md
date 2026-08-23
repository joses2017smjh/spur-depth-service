# Perception lineage this repo implements

The cutter does not care which paper coined the layer. It cares about
centimetres on a dormant spur. The stack below is what 2024–2026 orchard
robotics actually runs, mapped onto code in `spur_depth/pipeline/`.

## Detect (SSD → YOLO)

Liu et al., SSD (ECCV 2016) made single-shot boxes the default. Fruit and
trunk papers still cite it; head-to-heads on apple trees (Agriculture 2025)
show YOLO well ahead of SSD (91% vs 47% mAP on a small orchard set). 2025–26
pruning systems detect trunks with YOLOv5n+SENet or YOLOv9c, then segment
uprights (YOLOv8n-seg + DSConv) and read length/angle off the mask.

This repo does not ship a trained YOLO. It ships the **labels those heads
train on**: connected-component boxes from the Blender trunk / box masks
(`detect.boxes_from_mask`). Same output tensor a SSD/YOLO head would emit:
`(x1,y1,x2,y2,score,cls)`.

## Flow (FlowNet → RAFT)

Teed & Deng, RAFT (ECCV 2020) is still the flow model orchard visuomotor
work calls. Jain, Grimm, Lee (ICRA 2025, OSU CoRIS) put wrist-cam RAFT into
a PPO pruning policy and showed it beats a depth camera on thin dormant
wood, and transfers sim→real *because* it never trusted RGB photorealism.
You et al. (2025 review) pair FlowNet2 with pix2pix for branch masks.

`Optical_flow/` in this dataset is stereo RGB, not precomputed flow.
`pipeline.flow.dense_flow` computes it: torchvision RAFT when the weights
are present, OpenCV Farneback in CI.

## Metric depth (Zoe / DA → DA2-ft → RGB+D refine)

Yang et al., Depth Anything / V2 (CVPR 2024+). OrchardDepth++ (IROS 2025)
adds binned KL-flood so a monocular net can eat mixed camera / lidar
distributions. Prompt Depth Anything and Depth Any Camera (CVPR 2025) push
metric heads across intrinsics.

This service is that pipeline, already trained:

- Engine A — DA2 ViT-L fine-tune, ~0.060 m trunk RMSE, `POST /predict`
- Engine B+C — frozen DINOv2 + depth side-branch, 6 views,
  **0.0445 ± 0.0057 m**, `POST /predict/group`

## Reconstruct (MVS → back-project what you already measured)

MSA-MVSNet and classical RANSAC+ICP (BRANCH dataset, walnut ToF papers)
turn multi-view RGB into a cloud. We already have metric depth and Blender
`K`, `T_wc` on every frame. `pipeline.reconstruct.unproject` is the fusion
step those nets emit after depth regression — no second network, no invented
scale.

## Sensors (do not pick a winner)

AgriNav-Sim2Real (2025) ships RGB + depth + LiDAR + IMU + GPS in sim and
ZED2i + NIR + IMU in the field. Row-crop work bitwise-ORs segmentation
across time and gates it with a depth threshold. `pipeline.sensors.fuse_depths`
is inverse-variance fusion: each map brings a variance, the output is the
precision-weighted mean. Lidar / stereo / DA2 are extra maps, not comments.

## Sim → real (two checks, not a painted texture)

Three live strategies:

1. **Never trust RGB.** RAFT policies (ICRA 2025).
2. **Paint the sim.** DT/MARS-CycleGAN + YOLOv8 (JFR 2025).
3. **Measure a known object.** The 30 cm box this renderer already plants.

`pipeline.sim2real` does (3) plus appearance stats (luminance, focus,
saturation) against `baseline_stats.json`. No proxy is metric accuracy.
The box is.

## What this is not

A trained YOLO, a TensorRT millisecond, or a real-orchard RMSE. Those are
the next weights, the next container, and a field campaign — not a sentence
in this file.

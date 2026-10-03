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

Labels are still connected-component boxes from the Blender trunk mask
(`detect.boxes_from_mask`): `(x1,y1,x2,y2,score,cls)`. `pipeline.train_yolo`
fits an in-repo YOLO-nano (1 class, stride 16, 3 k-means anchors) on those
boxes without copying the 1080p frames. That is not Ultralytics YOLOv8n.
Checkpoint `weights/yolo_nano.pt` exists; val F1@IoU0.5 is **0.035** — not a
field detector. TinyUNet on 100 trees reaches val IoU 0.930; DA2 on predicted
masks is still 0.112 m vs 0.031 m on GT.

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

Correction (2026-10-02): with the render camera's intrinsics, the `box_mask`
region back-projects to a ~7 cm object fixed to the camera at z = 0.30 m,
not a 30 cm box, so `BOX_TRUE_M = 0.30` matches nothing in this data and the
box-anchor scale is not a metric check. `docs/readme/pipeline_report.json`
also fed the tree mask to it. Treat earlier box-anchor numbers as void.

## What this is not

A COCO YOLOv8n, a TensorRT *end-to-end* millisecond, a GPU-docker number
from a clean machine, or a 4-bark 24k-frame orchard. One bark is on disk:
100 trees, 6000 DA2 frames, hold-out affine **0.0325 m** on `00042`+`00065`.
Field RMSE on predicted boxes is still ~2 m — that is a detector problem,
not a missing-CSV problem.

## Branch perception (detect, localize, connect): what was adopted and why

The cutter needs a cut point and the branch axis there (Jain, Grimm, Lee,
ICRA 2025, [arXiv 2507.23015](https://arxiv.org/abs/2507.23015): success is
within 5 cm with pointing and perpendicularity within 30 deg; their 2026
hybrid-RL follow-up is [arXiv 2609.24906](https://arxiv.org/abs/2609.24906)).
Structural errors matter more than boundaries for pruning (Wang, Jain, He,
Grimm, Todorovic, CVPRW 2025, MFO dataset), so the module outputs a tree graph
and is scored on it (`docs/BRANCH_PROTOCOL.md`).

- **Image first, then lift.** At 1.4-2.6 m a D435-class sensor has
  sigma_z ~ 1.5 cm, five times a spur's radius, and loses wood under ~3
  sensor px. Point-cloud skeletonizers (Smart-Tree, IbPRIA 2023; DiffTS,
  ICCV 2025; AdTree; PC-Skeletor/CherryPicker, CVPRW 2023) assume dense clean
  clouds. The OSU follow-the-leader pipeline (You et al.,
  [arXiv 2309.11580](https://arxiv.org/abs/2309.11580)) skeletonizes in the
  image and lifts with depth, radius = z * r_px / f; `assemble.py` does the
  same and adds a depth-jump test so image crossings are not junctions.
- **Full resolution.** Spurs are 2-5 px wide at 1920x1080. Graph transformers
  (Relationformer, ECCV 2022; TreeFormer, WACV 2025; PlantPose, IJCV 2026) and
  ViT-patch segmenters work at 256-512 px. BranchNet keeps a ResNet-34 U-Net
  decoder and a stem at full resolution.
- **Topology-aware training.** Skeleton Recall (Kirchhoff et al., ECCV 2024)
  costs ~8% time; clDice as a loss costs +88% (Shit et al., CVPR 2021), so
  clDice-style quantities stay metrics. TopoMortar (BMVC 2025) finds strong
  augmentation does much of the rest.
- **Connect as a tree projection.** TreeFormer reports most of its topology
  gain from a minimum-spanning-tree step (TOPO-F1 0.708 -> 0.867 at test time
  alone, 0.870 with 98 h of constrained training). `assemble.py` solves a
  minimum-cost arborescence (Edmonds) over 3D attachment candidates with a
  botanical prior, as You et al. use semantic priors for cherry skeletons
  ([arXiv 2103.02833](https://arxiv.org/abs/2103.02833)). Dense direction
  fields (ViNet, CEA 2023; Sat2Graph, ECCV 2020; OpenPose PAFs) are the next
  edge scorer to add.
- **Sensor plus network depth.** Prompt Depth Anything (CVPR 2025) and Camera
  Depth Models ([arXiv 2509.02530](https://arxiv.org/abs/2509.02530), models
  for D435/D455/L515) use raw sensor depth as a metric prompt for a monocular
  network. `depth_fusion.py` is the closed-form version: Huber-fit DA2-ft to
  the sensor where both agree, keep the sensor there, DA2-ft elsewhere.
- **Sensor model.** Intel's D400 RMS error z^2 * subpixel / (f * B) at 848 px
  gives sigma_z = 0.0038 z^2, matching ~0.75% measured at 2 m.
- **Baselines in the field.** YOLOv8/Mask R-CNN detectors (Sapkota, Ahmed,
  Karkee 2024: P 0.90 / R 0.95 on dormant trunk/branch) and mask + depth + PCA
  diameters (Ahmed et al. 2023, 2.08 mm RMSE) give parts without topology;
  single-image hierarchy inference reaches ~80% F1 (Albaroudi et al., JRM 2026).
  Foundation segmenters (SAM 2/3) degrade as objects get more tree-like
  (WACV 2026), so they suit pseudo-labels for real video, not spur detection.

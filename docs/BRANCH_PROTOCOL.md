# Branch perception protocol (v1, fixed before test scoring)

Task: from one RGB-D frame, **detect** tree parts (trunk, scaffold branch,
shoot, spur), **3D-localize** them as metric axis polylines with radii, and
**connect** them into a parent/child tree with junction points. The pruning
output is a cut point per non-trunk part, 1.5 cm above its junction, with the
axis direction there (what a cutter needs to close perpendicular to the wood).

Scorer: `spur_depth/branches/metrics.py`. Harness: `scripts/eval_branches.py`.

## Data and splits

Blender renders of L-Py Envy trees, `bark_brown_02`, rigs `box` + `box_cam1-4`,
6 heights, stereo left and right: 60 frames per tree, 1920x1080.

| split | trees | use |
| --- | --- | --- |
| test | `lpy_envy_00042`, `lpy_envy_00065` (paper hold-out) | scored once, 120 frames |
| val | `00001`, `00009`, `00015`, `00041` | every threshold and setting is tuned here |
| train | the other 94 trees | network training only |

Ground truth (`spur_depth/branches/gt.py`) comes from the L-Py metadata the
renderer used, placed with the render camera's intrinsics (f = 1493.3 px, see
the intrinsics correction in the README). Each GT axis sample (1 cm spacing)
is *visible* when it projects inside the image onto its own part's pixels.
Parts need at least 25 labelled pixels and 3 visible samples to be scored.

## Methods

| method | classes from | notes |
| --- | --- | --- |
| `oracle` | GT class map | ceiling for graph assembly + depth |
| `tinyunet` | existing `weights/trunk_unet_100tree.pt` (256x256 tree mask, upsampled NEAREST as deployed) | classes from radius/length, which leans on L-Py regularities (10 mm vs 3 mm radii, 0.2 m spurs) that will not transfer; its pixels carry no classes, so only wood IoU applies; it was trained on the val trees (held out only 00042/00065), so its val rows are in-sample and never used for selection |
| `branchnet` | new full-resolution RGB-D network | trained on the 94 train trees |

All three use the same graph assembly (`spur_depth/branches/assemble.py`) with
one configuration chosen on val. Candidates (fixed before BranchNet's val
predictions exist): A defaults; B = A with `orphan_cost=1.2, junction_bonus=0`;
C = B with `depth_percentile=50`; D = C with `attach_tol_m=0.07`. The winner
maximises the mean of skeleton F1 @ 2 cm, edge F1 and cut F1 (Jain) for
`branchnet` with `fused` depth on the val trees; ties go to the earlier letter.
If BranchNet's val predictions are not available when scoring starts (the GPU
job is queued behind cluster limits), the same rule is applied to `oracle`
classes instead, and that choice is kept for every method. BranchNet's
checkpoint is the one with the best val tree-class mIoU during training
(sensor depth input).

## Depth sources

| source | what it models |
| --- | --- |
| `gt` | rendered depth: a perfect RGB-D camera |
| `sensor` | D435-style stereo depth (`depth_noise.py`: sigma_z = 0.0038 z^2 (Intel D400 formula at 848 px), thin wood dropped or bled to background, flying pixels, holes) |
| `da2` | DA2-ft monocular metric depth (RGB only) |
| `fused` | `sensor` kept where it agrees with Huber-aligned DA2-ft, DA2-ft elsewhere (`depth_fusion.py`); classifiers still see raw `sensor` |

## Metrics (micro-averaged over frames)

| metric | definition |
| --- | --- |
| skeleton P/R/F1 @ 1, 2, 5 cm | predicted axis samples (1 cm) within tau of GT axes of visible parts / visible GT samples within tau of a predicted axis; distances to axes densified to 2 mm |
| part P/R | one-to-one Hungarian matching on symmetric 3D overlap (3 cm), score >= 0.3; an unmatched prediction lying mostly on an unscored GT part is don't-care |
| part axis angle, radius error | matched pairs, principal directions of the overlapping portions |
| edge P/R/F1 | each predicted part maps to the GT part >= 50% of its samples lie on; an edge is correct when child and parent map to a GT child and its GT parent, each GT edge credited once; edges inside one GT part are neutral and edges whose child lies on an unscored part are don't-care; recall over *observable* GT edges (both parts scored, child base and parent near the junction visible) |
| edge P/R/F1 strict | same with the one-to-one matches (also charges parent fragmentation) |
| floating rate | matched parts with an observable GT parent edge but no predicted parent |
| junction error | predicted vs GT attach point for strictly correct observable edges |
| cut recall / precision | per tier, one-to-one assignment maximising successes between predicted cut points and GT cut points of scored non-trunk parts; **success = within 5 cm and axis within 30 deg** (Jain, Grimm, Lee, ICRA 2025); strict tier 2 cm and 15 deg (the Isaac cutter's 15 deg perpendicularity); recall over GT cuts with an observable edge; a prediction succeeding only on a non-observable GT cut is don't-care |
| IoU per class, mIoU | pixel classes, GT ignore pixels excluded |

## Primary metrics

1. Skeleton F1 @ 2 cm.
2. Edge F1 (geometric).
3. Cut recall and cut F1 at the Jain criterion.
4. Per-class IoU and tree-class mIoU (methods that predict classes).

## Comparisons claimed

1. `branchnet` vs `tinyunet` on the primary metrics, for every depth source.
2. `fused` vs `sensor` depth with `branchnet`.
3. `oracle` as the ceiling of the assembly.

Each summary records the scorer files' sha256, the commit and its cleanliness, the
assembly config, the frame-key hash and the prediction checkpoint; rows carry a
fingerprint and are reused only when it matches, and a summary with a missing
(method, depth, frame) row is marked incomplete. Test scoring refuses uncommitted
scorer files and any frame subsetting.

Two test trees and 120 correlated frames: results are reported per tree as
well as pooled, with no significance claims. Anything changed after the first
test scoring is a disclosed bug fix, never a retuned threshold.

## Disclosures (2026-10-02 run)

- Geometry checks only, no branch metric: the intrinsics fix and the GT
  labelling were validated on test-tree frames (`lpy_envy_00042` box_cam1
  shot03_l, box and box_cam1 shot01 l/r) before the protocol existed.
- Test-frame metrics seen before test scoring: figure dry runs on
  `lpy_envy_00065` box_cam1 (6 heights, left camera) between 20:18 and 20:45
  PDT printed and plotted per-frame skeleton F1, edge F1 and cut recall for
  frame shot03_l (four depth sources, ground-truth and TinyUNet classes), first
  with the assembly at `3060ef8` (seen 20:29), then at `8c1f54f` (depth panel,
  seen about 20:45). The assembly changes in `8c1f54f` came from the third
  blind reviewer's val-only findings, were checked by an old/new A/B on 12 val
  frames, and the config selection was re-run on val afterwards (C again).
  No other test output was read before `run_C/summary.json` at 21:25.
- BranchNet (comparison 1) was trained and scored after the oracle and TinyUNet
  test rows had been read (21:25 PDT). Its first run (Slurm 21520501, `e23b40f`)
  hung at the start of epoch 2 (data-loader fork deadlock) and was cancelled
  with the user's approval after 65.7 min. The rerun (Slurm 21523238, `3fa8420`)
  trained on the train trees only and picked its checkpoint by val mIoU; it was
  scored once (rows 00:39-00:51, summary 01:01 PDT, 2026-10-03) from the frozen
  clone `776c36a` with config C
  (fixed at 20:58, before any test row existed), into its own directory, so
  the earlier rows are unchanged.

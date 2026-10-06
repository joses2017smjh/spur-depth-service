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

# v2 additions (2026-10-03, fixed before any v2 test row)

The splits, ground truth, scorer and assembly config C are unchanged from v1:
`git diff 776c36a -- spur_depth/branches/metrics.py` is empty, and the new
assembly option defaults to the v1 behaviour (GT and default predicted cuts
checked byte-for-byte on four val frames). Every v2 comparison is chosen on
the val trees, then scored once on the test trees from a frozen clone, into
its own directory, with the v1 fingerprinted rows and integrity checks.

## New methods and their val selection rules

| item | candidates | val rule (ties to the earlier candidate) |
| --- | --- | --- |
| fitted cut axis (`cut_axis="fit"`): a predicted part's cut direction is the principal direction of its axis between arc lengths lo and hi above the junction, past the bend that thinning puts into a skeleton at a junction; GT cuts keep the local tangent | tangent (v1); fit 3-8 cm; fit 2-10 cm; fit 3-15 cm | highest strict cut recall (2 cm, 15 deg), `branchnet` + `fused`, all 240 val frames |
| `cdm` depth: Camera Depth Model refinement (arXiv 2509.02530, ICLR 2026) of the same simulated D435 frame, by `scripts/refine_depth_cdm.py`; lift only, as for `fused` (the classifier sees raw sensor depth); weights CC-BY-NC-4.0, trained indoors | round 1: checkpoint (d435, l515, base) x input size (518, 1036), after the channel order and fp16 were checked. Round 2, registered after round 1 failed (best: 3% of wood pixels within 2 cm, raw sensor 66%): `base` at 518 with invalid sensor pixels outside the dilated predicted wood (BranchNet, as `fused` uses) set to a far depth: 15 px and 20 m, 15 px and 8 m, 40 px and 20 m | largest share of GT wood pixels within 2 cm of GT depth, every 5th val frame; round 2 replaces round 1 only if its winner beats round 1's; no accuracy is computed on test frames |
| BranchNet fine-tune (`branchnet` with the new run's predictions): warm start from run 21523238's best.pt (epoch 20, val mIoU 0.784), decoder lr 2e-4 (encoder x0.33), 100 warm-up iterations, seed 1, 80 min (Slurm 21532184, commit `eabe197`) | one run | checkpoint by val tree-class mIoU, as in v1. Its test rows are scored whatever its val mIoU. The v1 curve had flattened with the learning rate fully decayed, so a small change is expected |
| multi-view fusion (`<method>_mv`, `scripts/eval_branches_multiview.py`, `spur_depth/branches/multiview.py`): one group is one rig of one tree, 6 heights x 2 eyes = 12 frames; each frame's graph (config C + the chosen cut axis) is moved to the world frame with its logged pose, parts are associated across frames by 3D overlap, merged (per-cm median along the longest member) and connected by parent votes and a maximum spanning arborescence; each frame is scored on its view of the fused graph, culled by that frame's own predicted wood mask (never GT) | association tolerance 3 or 5 cm x minimum views 1 or 2 x view `anchored` (only parts the frame itself detected) or `seen` (any fused part on the frame's wood) x axis smoothing 0 or 2 cm (16 configs, base extension on) | highest mean of skeleton F1 @ 2 cm, edge F1 and cut F1 (Jain), `branchnet_mv` + `fused`, all 20 val groups, with the cut axis chosen above. Multi-view is scored on test only if that config beats single-frame on the same mean on val (same frames, same assembly); otherwise it is reported as a val-only result |

## Real images and a robot-geometry proxy (no test-tree tuning either)

- **MFO cherry UFO "Labelled Data"** (MFO dataset, CVPRW 2025; real
  640x480 RGB, labelme polygons), zero-shot BranchNet with no depth input
  (valid = 0, seen in 10% of training crops). Classes: `leader` to trunk,
  `sidebranch` to branch (BranchNet's branch and shoot merged), `spur` to spur,
  `other` ignored (wood for the binary score), wires and unlabelled to
  background. The resize factor (x1, x1.6, x2.4) is chosen on the MFO val split
  (videos 164-173) by 3-class mIoU, then reported on the MFO test (174-184) and
  train (89-163) splits. No MFO image is committed: the data carry no licence.
- **Isaac cut proxy** (`scripts/isaac_cut_proxy.py`, CPU only): each predicted cut
  paired with a GT cut (the scorer's Jain-tier assignment) drives the UR5e
  pruner geometry from a frozen clone of isaac-sim-pruning-workflow, posed by a
  fixed rule (approach along the camera ray made perpendicular to the perceived
  axis, closing axis = approach x axis). Success is judged against the true
  branch and the other true wood. The truth-segment length and the pose rule are
  fixed on val with a perfect-perception control. It is a geometric proxy, not
  an Isaac result.

## Comparisons claimed (v2)

1. Fitted vs tangent cut axis, every method and depth source.
2. `cdm` vs `fused` vs `sensor` depth, `branchnet` and `oracle`.
3. Fine-tuned vs v1 BranchNet.
4. Multi-view vs single-frame (`branchnet` with `sensor` and `fused`, `oracle` with `fused`).
5. MFO zero-shot pixel scores, reported as absolute numbers. The paper's
   adapted models (MIC 47.6, SGDR 56.2 mIoU, trained on unlabelled real images)
   are context, not a baseline under the same protocol.
6. Proxy cut success with tangent and fitted axes, plus the perfect-perception ceiling.

## Disclosures (v2, before any v2 test row)

- Val only: the cut-error diagnostic that motivated the fitted axis (every 6th
  val frame); a partial read (62 of 240 frames) of the cut-axis selection run
  while it was still running; and four multi-view smoke runs on group
  `lpy_envy_00001/box_cam1` that led to the `anchored` view, the smoothing
  option and an `extend_base` switch (left on: switching it off did not help).
- CDM round 1's val table (every setting far worse than the raw sensor on wood)
  was read before round 2 was registered; two val frames' wood errors were
  printed by smoke runs before round 1's rule was written.
- The MFO scoping agent looked at GT overlays of four labelled frames (MFO test
  174_15, 177_34, 184_20 and one RozaCloudyAfternoon frame) and measured label
  widths on the 15 MFO test frames; nothing was tuned on them.

## v2 val outcomes (fixed 2026-10-04, before any v2 test row)

| item | val outcome | evidence |
| --- | --- | --- |
| cut axis | `fit` 3-15 cm wins: strict cut recall 0.189 vs tangent 0.134 (`branchnet` + `fused`, 240 frames). Test config = C + `{"cut_axis": "fit", "cut_fit_lo_m": 0.03, "cut_fit_hi_m": 0.15}` | `bench/results/branches_val_cut_axis_2026-10-03.json` |
| CDM depth | round 1 best `base`@518: 3.0% of wood pixels within 2 cm; round 2 winner (15 px, 20 m) 7.7%, so round 2 is the `cdm` source; raw sensor 65.7%, `fused` 68.5%. Expected to lose to `fused`; scored on test as registered | `/nfs/hpc/share/sanchej7/spur-branch-eval/depth/cdm_skyfill_val_selection.json`, `.../cdm_val_selection.json` |
| fine-tune | Slurm 21532184: 37 epochs in 79.4 min, best val tree-class mIoU 0.801 at epoch 30 (v1 run 0.784); 81.7 GPU-min of the 110 approved | `.../spur-branch-runs/21532184/history.json` |
| multi-view | winner tolerance 3 cm, at least 2 views, `anchored`, 2 cm smoothing: criterion 0.505 vs single-frame 0.480 (skeleton F1 0.643 vs 0.603, edge F1 0.441 vs 0.427, cut F1 0.429 vs 0.410; strict cut recall 0.146 vs 0.189). It beats single-frame, so it is test-scored with fuse config `{"assoc_tol_m": 0.03, "min_views": 2, "view_mode": "anchored", "smooth_m": 0.02}` | `bench/results/branches_val_multiview_2026-10-03.json` |
| MFO zero-shot | resize x1 chosen on MFO val (3-class mIoU 0.097 vs 0.091 at x1.6 and 0.065 at x2.4), written before any MFO train or test frame was predicted | `/nfs/hpc/share/sanchej7/spur-realdata/results/mfo_selection.json` |
| Isaac proxy | truth segment 0.10 m (the control's isolated success is 1.0 at every length on the grid; the largest admissible length is kept) | `/nfs/hpc/share/sanchej7/spur-branch-eval/isaac_proxy/length_calibration.json` |

The `cdm` source of `scripts/eval_branches.py` now defaults to the round-2
directory. The fine-tune's predictions (`spur-branch-runs/21532184/pred`) are
scored as `branchnet` in a separate output directory, so they never share rows
with run 21523238's.

## v2 test scoring record (2026-10-04)

- Single-frame (`test_v2/fit`: oracle, TinyUNet, BranchNet run 21523238; `test_v2/ft`:
  the fine-tune) and multi-view (`test_v2/mv`) rows were scored once from the
  frozen clone `adabaf8`, with config C + the fitted axis and the registered fuse
  config, into their own directories. The v1 rows are untouched.
- Review: two blind review agents (Isaac proxy; multi-view harness and the v2 scorer
  changes) were stopped by API rate limits before reading any code. Checks that ran
  instead: the multi-view test harness reproduces the val driver's rows for the selected
  config exactly (804 fields over the 12 frames of `lpy_envy_00001/box`); a main-session
  read of the proxy's graph-cache key (it includes the full assembly config, the
  prediction provenance and the build-path file hashes), its pairing (the scorer's Jain
  assignment, input for input), its test guard, and its unit tests (5 pass).
- Isaac proxy test runs: one output directory per configuration (tangent, fitted axis)
  and method, from a frozen clone, with the registered truth length (0.10 m).
- Test summaries were first read at 15:19 PDT on 2026-10-04, after every v2 test row
  existed. The README figures were rendered afterwards (`branch_cuts.gif`,
  `branch_multiview.gif` on `lpy_envy_00065` box_cam1, `branch_v2.png`); their scripts
  were tried on val tree `lpy_envy_00009` first. `branch_cuts.gif` was first rendered
  with fused depth, then re-rendered with rendered depth, which shows the axis change
  without depth error. That is a presentation choice made after seeing those test frames;
  the tables report every depth source.
- GPU: the fine-tune used 81.7 of the 110 approved GPU-minutes (dgxh). Local 2080 Ti:
  CDM about 25 min (selection, sky-fill round and inference), MFO about 8 min.

# v3: synthetic UFO trees (registered 2026-10-04, before any UFO frame was rendered)

Why: on real MFO cherry frames (UFO architecture) BranchNet scored 3-class mIoU 0.12 zero-shot
after training on Envy renders only. `Computer_Vision/trees` also holds 100 L-Py UFO trees
(`lpy_ufo_*`) that were never rendered. Budget approved by the user: one smoke tree, then 39
more if the smoke gate passes (about 1,400 GPU-min of rendering), and 240 GPU-min of training.

## Data
- Generator: `Computer_Vision/Dataloader/generate_tree2.py` (sha256 `2dedbbc1...`), run as for
  the Envy set (bark `bark_brown_02`, rigs box + box_cam1-4, 6 heights x stereo l/r, 60 frames,
  DA2-ft depth), with one change: `scripts/blender/generate_tree2_ufo.patch`. The generator's
  hierarchy walk kept only `branch_`, `nontrunk_`, `trunk` and `spur_` children, so it would
  silently drop the UFO laterals (`tertiarybranch_`, 0.26 m, 3 mm, about 18 per tree) and
  their spurs. Launcher: `scripts/run_render_ufo.sh`. Output:
  `/nfs/hpc/share/sanchej7/spur-ufo/full_spur_ufo`, with one manifest per tree.
- Splits, by tree id: train `lpy_ufo_00000`-`00027` (28 trees), val `00028`-`00031` (4),
  test `00032`-`00039` (8). The smoke tree `00000` is a train tree. The Envy splits are unchanged.
- Classes: L-Py `trunk` (the horizontal cordon) -> trunk, `branch` (the uprights) -> branch,
  `tertiarybranch` -> shoot (same scale as Envy's 0.3 m `nontrunk` shoots), `spur` -> spur.
  The ground-truth walk (`gt.reachable_parts`) follows the patched generator.

## Smoke gate (tree 00000; viewing it is allowed, it is a train tree)
1. All 60 frames and all 60 DA2-ft maps are written.
2. Ground-truth depth lies on the cylinder surfaces: median residual <= 2 mm, as for Envy.
3. The tree is framed: at least 50 frames carry at least 5,000 labelled tree pixels.
If any check fails, nothing more is rendered and the user is asked.

## Evaluation (fixed before any UFO prediction exists)
- Models: BranchNet Envy-only (run 21532184) vs BranchNet Envy + UFO (warm start from 21532184,
  trained on the 94 Envy and 28 UFO train trees, checkpoint by mean tree-class mIoU over the
  4 Envy and 4 UFO val trees). Assembly: config C + the fitted cut axis.
- Synthetic: the 8 UFO test trees (new) and the 2 Envy test trees (regression check), every
  primary metric, depths gt, sensor, da2 and fused, with the GT-class ceiling.
- Real MFO cherry: same frames, splits and pre-processing as v2, with the resize factor
  re-chosen on MFO val for each model. Two class mappings: the v2 mapping (leader -> trunk,
  sidebranch -> branch + shoot), kept for comparison with 0.12, and the UFO mapping (MFO
  `leader` is the upright: predicted trunk + branch -> leader, shoot -> sidebranch,
  spur -> spur). Primary: 3-class mIoU under the UFO mapping, plus binary wood IoU, which
  needs no mapping.

## v3 training run (registered before any UFO frame was labelled)
- One sbatch job (`scripts/run_train_branches.sh`, frozen commit): `--ufo --init
  spur-branch-runs/21532184/best.pt --lr 2e-4 --warmup 100 --seed 2 --val-frames 96`,
  80-150 min of training plus prediction of the Envy and UFO val and test trees, inside
  the approved 240 GPU-min. Training data: the 94 Envy and 28 UFO train trees. The
  checkpoint is chosen by the mean of the Envy-val and UFO-val tree-class mIoU (48 val
  frames each), so each domain counts equally.
- Smoke-render record: the first smoke job (21549311, about 4 GPU-min) rendered only the
  front rig because the patched generator copy could not import its sibling
  `move_camera.py`. The launcher now copies the siblings (hash-checked), and the gate is
  applied to the second job's output.
- MFO rescoring (`scripts/rescore_mfo_mapping.py`) reproduces v2's numbers from the saved
  confusion counts (val/test/train 0.097/0.123/0.112). Under the UFO mapping the Envy-only
  model scores 0.071/0.092/0.073, with the same wood IoU (0.171/0.236/0.189).
- Smoke gate (job 21549346, 21.8 GPU-min on an RTX 8000): pass. 60 frames, 60 DA2-ft maps,
  median GT surface residual 0.099 mm, 52 of 60 frames with at least 5,000 labelled pixels
  (`bench/results/branches_ufo_smoke_gate_2026-10-04.json`, `scripts/check_ufo_smoke.py`).
  Two findings from viewing the smoke tree (a train tree):
  - No trunk pixels in any frame. The UFO cordon lies at ground level where the generator
    places the tree, so UFO frames carry uprights (branch), laterals (shoot) and spurs only.
    Trunk IoU on UFO trees is therefore undefined, and mIoU uses the classes present.
  - The top camera height sees no tree, because UFO trees are about 2.4 m tall.
  Neither changes a registered rule. The other 39 trees render unchanged.
- Execution (submitted 2026-10-04 ~21:45 as Slurm dependencies, so every step runs
  without anyone looking at intermediate results; rows and summaries come from frozen
  clone `081b76b`, config C + fitted axis):
  - renders: array 21549460 (39 trees, 6 at a time);
  - UFO label cache: 21549499;
  - Envy + UFO training: 21549500 (150 min plus prediction);
  - Envy-only predictions on the UFO val/test trees: 21549501;
  - MFO for the new model: 21549502 (`spur-realdata/v3_envy_ufo`);
  - test scoring: `test_v3/ufo_envyonly` (GT classes and Envy-only BranchNet, UFO test)
    21549512/21549513, `test_v3/ufo_envyufo` 21549514/21549515, `test_v3/envy_envyufo` (Envy test,
    regression check) 21549516/21549517.
  - GPU, about 180 of the 240 approved minutes: training ~165, Envy-only predictions ~6,
    MFO ~8. Renders: ~22 GPU-min per tree.
- Renders done (2026-10-05 01:39): 40 trees x 60 frames, all with DA2-ft maps and the same
  patch (manifests in `spur-ufo/full_spur_ufo/manifests`). GPU: 1,285 min for the 39 trees
  (RTX 8000, A40 and V100 nodes; 33 min per tree on average) + 22 for the smoke tree + 4
  for the failed first smoke = 1,311 of the ~1,400 approved. Label cache: 2,400 UFO frames,
  no errors. Training split: 5,640 Envy + 1,680 UFO train frames; val 240 + 240; test
  120 + 480; no val or test tree in training (checked).
- Clarification (2026-10-05 01:52, after reading training epoch 1 on val, before any test row):
  the tree-class mIoU in training selection and in the scorer averages over the classes with
  a non-empty union (ground truth or prediction), the v1/v2 convention. UFO frames have no
  trunk pixels, so a trunk predicted on them scores trunk IoU 0. That is a penalty for
  hallucinated trunk, not an undefined class. At epoch 1 UFO-val mIoU is 0.535 (trunk 0,
  branch 0.916, shoot 0.595, spur 0.627); over the three classes present it would be 0.713.
  Envy-val mIoU is 0.790. The selection score can jump at an epoch where UFO trunk
  predictions vanish entirely. Results report both the 4-class and the 3-class (branch,
  shoot, spur) UFO mIoU. Training job 21549500 runs on an H100 MIG 3g.40gb slice.
- First v3 test result read at 02:17 PDT, 2026-10-05: `test_v3/ufo_envyonly`, the GT classes
  and the Envy-only BranchNet on the 8 UFO test trees (complete, clean `081b76b`;
  `bench/results/branches_test_v3_ufo_envyonly_2026-10-05.json`). The Envy + UFO model was
  still training (job 21549500); its rows, the Envy regression rows and the MFO run follow
  automatically.
- v3 results (all chained jobs completed by 04:23 PDT, 2026-10-05; read at 07:50):
  - Training 21549500: 49 epochs in 150 min, checkpoint epoch 46. Selection jumped at the
    epochs where UFO trunk predictions vanished (31 and 46); epoch 46 sits on the plateau
    (Envy-val 0.81, UFO-val 3-class 0.80, as at the last epochs). GPU: 154 min training +
    prediction, 4.5 Envy-only prediction, 1 MFO = about 160 of the 240 approved.
  - UFO test (8 trees), Envy-only vs Envy + UFO: 3-class pixel mIoU 0.49 -> 0.78 (branch
    0.59 -> 0.94, shoot 0.33 -> 0.66, spur 0.56 -> 0.74), but the graph barely moves.
    Rendered depth: skeleton F1 0.840 -> 0.833, edge F1 0.738 -> 0.750, cut F1 (Jain)
    0.585 -> 0.616. Fused depth: skeleton F1 0.468 -> 0.476 (GT-class ceiling 0.480).
  - Envy test: no regression; every graph metric within 0.005, mIoU 0.794 -> 0.798.
  - Real MFO: no gain. UFO-mapping test mIoU 0.092 -> 0.082, wood IoU 0.236 -> 0.221, and
    background called wood 31% -> 36%. The real-image gap is appearance (backgrounds,
    lighting, bark), not tree architecture.

# v4: orchard context (registered 2026-10-05, before any orchard frame was rendered)

Why: v3's UFO training raised UFO pixel mIoU from 0.49 to 0.78 but not real MFO (0.092 -> 0.082).
On real frames the model calls other rows and the ground wood: the renders show one tree on
bare ground, while real frames show a full row and further rows behind. The user asked for the
whole orchard, using measured row spacing and 3 more rows. Approved: 40 UFO + 30 Envy trees
in orchard context (about 3,900 GPU-min) and 240 GPU-min of training.

## Scene (`scripts/blender/orchard_context.py`, `generate_tree2_orchard.patch`, `CV_ORCHARD=1`)
- Target row: trees on both faces of the V, one per bay; 3 more rows behind the target row,
  each with posts and wires.
- Row spacing: Envy V-trellis 3.53 m (Prosser WA; row width 353 +- 3 cm, tree spacing
  142 cm, 7 wires 46 cm apart, canopy angle 75 deg, tree height 366 cm; Davidson et al. 2016
  as tabulated in Bhattarai et al., arXiv 2304.04919). UFO sweet cherry 3.05 m (WSU Roza
  farm, Prosser; 1.83 m within the row).
- Bays: Envy 3.93 m (the template's post bay; the L-Py Envy trees span 3.96 m). UFO 2.40 m
  (the L-Py UFO trees span 2.37 m, larger than Roza's 1.83 m spacing). Neighbour trees are
  drawn from the train split of the same kind.
- Labels: same-row trees share the target's mask index and match no target cylinder, so the
  ground truth marks them IGNORE (never trained on, either way). Trees, posts and wires of the
  other rows are background, as in MFO's real labels.
- Render keys `orchard_<tree id>` in `spur-ufo/full_spur_orchard`
  (`scripts/run_render_orchard.sh`); layouts in `orchard_layout/`.

## Splits (by tree id, fixed now)
- UFO: the v3 splits (train 00000-00027, val 00028-00031, test 00032-00039).
- Envy: train = the first 24 ids outside the held-out set {00001, 00009, 00015, 00041,
  00042, 00065}; val = the Envy val trees; test = the paper test trees 00042 and 00065.

## Smoke gate (tree `lpy_ufo_00000`, a train tree)
The v3 gate (60 frames and DA2 maps, median residual <= 2 mm, >= 50 frames framed), plus:
same-row neighbours appear as IGNORE in at least 30 frames, and rows behind are rendered
(the layout file lists all 4 rows). Measured GPU-min per tree is reported before the batch.

## Training (240 GPU-min)
One job: warm start from run 21549500 (Envy + UFO), `--ufo --orchard`, lr 2e-4, 100
warm-up iterations, seed 3, 150 min, val frames 144. Checkpoint by the mean over the three
domains (Envy, UFO and orchard val) of mIoU over the tree classes with ground-truth pixels
in that domain. This fixes v3's trunk artefact.

## Evaluation (fixed now)
- Primary: real MFO cherry, both class mappings, resize re-chosen on MFO val. v4 vs v3 vs
  the Envy-only model.
- Synthetic regression: single-tree UFO test (8) and Envy test (2), all primary metrics.
- Orchard test trees (8 UFO + 2 Envy in context): pixel IoU with IGNORE excluded, v3 vs v4.
  Graph metrics are reported but flagged: neighbour wood the model finds counts against
  precision there.
- Orchard smoke gate (job 21565423, `orchard_lpy_ufo_00000`, 38.9 GPU-min on a V100): pass.
  63 trees in the scene, 60 frames and 60 DA2-ft maps, median GT residual 0.098 mm, 50 of 60
  frames framed, same-row neighbours IGNORE in all 60 frames, all 4 rows in the layout
  (`bench/results/branches_orchard_smoke_gate_2026-10-05.json`, `scripts/check_smoke_gate.py`).
  Viewed on one frame: target uprights labelled, neighbours IGNORE, rows behind background.
- v4 execution (submitted 2026-10-05 ~10:08 as Slurm dependencies; rows and summaries from
  frozen clone `6cd4818`, config C + fitted axis): renders 21566499 (69 trees, 6 at a time)
  -> orchard label cache 21566535 -> training 21566536 (`--ufo --orchard`, init 21549500)
  and v3-model predictions on the orchard val/test trees 21566537 -> MFO 21566538 for the v4
  model (`spur-realdata/v4_orchard`). Scoring into `spur-branch-eval/test_v4/`
  (array/summary): `orchard_v3model` 21566539/40, `orchard_v4model` 21566541/42,
  `ufo_v4model` 21566543/44, `envy_v4model` 21566545/46.
- Storage cleanup (2026-10-05, with the user's approval): deleted the CDM refined depth maps
  (`spur-branch-eval/depth/cdm`, `cdm_skyfill`), the unused CDM d435/l515 weights, the hung
  first BranchNet run (21520501) and the `last.pt` files of finished runs. The CDM numbers,
  manifests and file hashes stay in the committed evidence; re-scoring `cdm` would first need
  `scripts/refine_depth_cdm.py` re-run (base checkpoint kept).
- v4 results (training and three scoring runs finished by 21:00 PDT, 2026-10-05; read at 22:00):
  - Training 21566536: 43 epochs in 150 min (A40), checkpoint epoch 43; val mIoU over the
    classes present: Envy 0.817, UFO 0.808, orchard 0.780.
  - Real MFO (primary), resize chosen on MFO val (x1.6). Under the UFO mapping, test mIoU went
    from 0.092 (Envy-only) and 0.082 (v3) to 0.121. Wood IoU 0.236 / 0.221 -> 0.333.
    Background called wood fell from 31% / 36% to 8%; labelled-wood recall fell from 0.75 to
    0.54. Leader IoU 0.206 -> 0.299, spur 0.016 -> 0.032. The gain holds on MFO val (0.071 ->
    0.093) and train (0.073 -> 0.113). Under the v2 mapping (leader = trunk) test falls to
    0.045, because v4 calls the uprights "branch", as the UFO renders teach it to.
  - Synthetic regression: UFO single-tree test pixel IoU branch/shoot/spur 0.94/0.66/0.74 ->
    0.95/0.68/0.76; Envy test trunk 0.91 -> 0.92. Graph metrics are within 0.016 (UFO
    skeleton F1 on rendered depth 0.833 -> 0.817; Envy 0.877 -> 0.883).
  - Orchard test, v4 model: pixel IoU trunk/branch/shoot/spur 0.91/0.92/0.51/0.71. Graph
    metrics are low, as flagged (skeleton F1 0.36 on rendered depth), because neighbour wood
    counts against precision.
  - Incident: the v3-model + GT-class orchard scoring (array 21566539) hit its 2 h limit with
    1,730 of 4,800 rows written (orchard frames take about 12 min for 8 assemblies). It was
    resubmitted unchanged (same frozen clone, rows reused by fingerprint) as 21594223 (60
    shards, 4 h) with summary 21594224. The original summary job 21566540 cannot run
    (DependencyNeverSatisfied).
- Orchard test (resubmitted scoring complete, clean `6cd4818`, read 2026-10-06 00:00): on 10
  orchard test trees (600 frames, rendered depth), v3 -> v4 -> GT classes:
  - pixel mIoU 0.23 -> 0.76 -> 1.00 (shoot 0.02 -> 0.51, spur 0.07 -> 0.71);
  - skeleton F1 @2 cm 0.01 -> 0.36 -> 0.81;
  - edge F1 0.04 -> 0.39 -> 0.73;
  - cut recall (Jain) 0.43 -> 0.59 -> 0.61.
  
  The v3 model calls nearly every pixel of an orchard frame wood (skeleton precision
  0.005). With fused depth even GT classes collapse (skeleton F1 0.05): DA2-ft fusion breaks
  on the rows behind the target. The stale summary job 21566540 was cancelled with the
  user's approval.
- README figures (2026-10-06, rendered after all v3/v4 test rows were read):
  - `branch_ufo_compare.gif`: UFO test tree `lpy_ufo_00035`.
  - `branch_orchard_compare.gif`: classes mode, orchard test tree `orchard_lpy_ufo_00035`.
  - `orchard_scene.png`: train trees only.
  - `branch_mfo.png`: numbers only, no MFO image; palette checked with the dataviz validator (ordinal blue ramp).

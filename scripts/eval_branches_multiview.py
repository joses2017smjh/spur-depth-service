"""Score multi-view fusion per frame, with the same scorer, GT and rows as eval_branches.py.

One fusion group is one rig of one tree: 6 heights x 2 stereo eyes = 12 frames,
the sweep a robot makes in front of a tree. Every frame of the group is
assembled exactly as in eval_branches.py, moved to the world frame with its
logged pose, fused (spur_depth.branches.multiview), and each frame is then
scored on its own view of the fused graph against its own ground truth.
Methods are named ``<method>_mv`` so their rows and summaries never mix with
single-frame rows.

python scripts/eval_branches_multiview.py --split val --methods branchnet --depths fused \
    --pred-dir RUN/pred --cfg '{...}' --fuse '{"view_mode": "anchored"}' --out OUT
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_branches as ev  # noqa: E402

from spur_depth.branches import gt as G  # noqa: E402
from spur_depth.branches.assemble import AssembleConfig, assemble, fit_cut_axes  # noqa: E402
from spur_depth.branches.dataset import frame_depth, load_cached_frame, read_depth_m  # noqa: E402
from spur_depth.branches.depth_fusion import fuse_sensor_mono  # noqa: E402
from spur_depth.branches.metrics import evaluate_frame, segmentation_counts  # noqa: E402
from spur_depth.branches.multiview import FuseConfig, fuse, to_world, view_graph  # noqa: E402
from spur_depth.branches.tree import IGNORE  # noqa: E402

SELF = Path(__file__).resolve()


def groups(split: str) -> list[tuple[str, str]]:
    trees = G.PAPER_TEST if split == "test" else G.VAL_TREES
    return [(t, r) for t in trees for r in G.RIGS]


def frame_pose(ref: G.FrameRef, cache: Path) -> np.ndarray:
    meta = json.loads((cache / ref.tree / ref.rig / f"{ref.stem}.meta.json").read_text())
    return np.asarray(meta["T_wc"], dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--methods", nargs="+", default=["branchnet"], choices=("oracle", "branchnet"))
    ap.add_argument("--depths", nargs="+", default=["fused"], choices=ev.DEPTHS)
    ap.add_argument("--cache", type=Path, default=ev.CACHE)
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--cdm-dir", type=Path, default=ev.CDM_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cfg", type=str, default="{}", help="JSON AssembleConfig overrides")
    ap.add_argument("--fuse", type=str, default="{}", help="JSON FuseConfig overrides")
    ap.add_argument("--shard", type=int, default=0, help="over (tree, rig) groups")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg))
    fcfg = replace(FuseConfig(), **json.loads(args.fuse))
    if "branchnet" in args.methods and args.pred_dir is None:
        ap.error("--methods branchnet needs --pred-dir")
    if args.split == "test":
        state = ev.git_state(ev.REPO)
        mine = subprocess.run(
            ["git", "-C", str(ev.REPO), "status", "--porcelain", "--", str(SELF)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if state["dirty"] or mine:
            sys.exit(f"refusing test scoring with uncommitted scorer files:\n{state} {mine}")
    prov = ev.provenance(args, cfg)
    prov["fuse_config"] = asdict(fcfg)
    prov["multiview_script_sha256"] = ev._sha(SELF)
    # Fingerprints, frame lists and summaries come from eval_branches.py with "<method>_mv".
    sargs = argparse.Namespace(**vars(args))
    sargs.methods = [f"{m}_mv" for m in args.methods]
    sargs.every, sargs.limit = 1, 0
    if args.summarize_only:
        return 0 if ev.summarize_dir(sargs, prov)["complete"] else 1

    out_frames = args.out / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for gi, (tree, rig) in enumerate(groups(args.split)):
        if gi % args.n_shards != args.shard:
            continue
        refs = [r for r in G.list_frames([tree]) if r.rig == rig]
        for method in args.methods:
            for source in args.depths:
                name = f"{method}_mv"
                fp = ev.fingerprint(prov, name, source)
                paths = [
                    out_frames / f"{name}__{source}__{r.key.replace('/', '__')}.json" for r in refs
                ]
                if all(
                    p.is_file() and json.loads(p.read_text())["meta"]["fingerprint"] == fp
                    for p in paths
                ):
                    continue
                per = []
                ts = time.time()
                for ref in refs:
                    gtg, gt_cls, K = ev.gt_graph(ref, args.cache)
                    frame = load_cached_frame(ref, args.cache, skel=False)
                    net_source = "sensor" if source in ev.LIFT_ONLY else source
                    cls, conf, tweak, per_class = ev.predicted_classes(
                        method, ref, net_source, gt_cls, frame, args.pred_dir, None
                    )
                    wood = (cls > 0) & (cls != IGNORE) if method != "oracle" else gt_cls > 0
                    if source == "fused":
                        depth, _ = fuse_sensor_mono(
                            frame_depth("sensor", frame, ref), frame_depth("da2", frame, ref), wood
                        )
                    elif source == "cdm":
                        depth = read_depth_m(
                            args.cdm_dir / ref.tree / ref.rig / f"{ref.stem}.depth.png"
                        )
                    else:
                        depth = frame_depth(source, frame, ref)
                    pred = assemble(cls, depth, K, conf=conf, cfg=replace(cfg, **tweak))
                    seg = segmentation_counts(np.where(cls == IGNORE, 0, cls), gt_cls, per_class)
                    per.append((ref, gtg, K, frame_pose(ref, args.cache), wood, pred, seg))
                fused = fuse([to_world(x[5], x[3]) for x in per], fcfg)
                if cfg.cut_axis == "fit":
                    fit_cut_axes(fused, cfg.cut_fit_lo_m, cfg.cut_fit_hi_m)
                for fi, ((ref, gtg, K, T, wood, _, seg), row_p) in enumerate(zip(per, paths)):
                    vg = view_graph(fused, T, K, wood, fcfg, frame=fi)
                    row = evaluate_frame(vg, gtg)
                    row["seg"] = seg
                    row["meta"] = {
                        "key": ref.key,
                        "tree": ref.tree,
                        "method": name,
                        "depth": source,
                        "fingerprint": fp,
                        "assemble_s": round((time.time() - ts) / len(refs), 3),
                        "n_parts_pred": len(vg.parts),
                        "fused_tracks": len(fused.parts),
                    }
                    ev._write_atomic(row_p, json.dumps(row))
                print(f"[{tree}/{rig}] {name}/{source} {time.time() - t0:.0f}s", flush=True)
    if args.n_shards == 1:
        return 0 if ev.summarize_dir(sargs, prov)["complete"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

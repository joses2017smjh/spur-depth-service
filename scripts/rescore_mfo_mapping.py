"""Re-score saved MFO confusion counts under a class mapping (protocol v3), no prediction.

scripts/eval_real_mfo.py saves, per frame and resize factor, the confusion counts of GT
{background, leader, sidebranch, spur, other, nonbranch} x predicted {background, trunk,
branch, shoot, spur}. Mappings:

  v2   leader -> trunk, sidebranch -> branch + shoot, spur -> spur (Envy semantics)
  ufo  leader -> trunk + branch (MFO's leaders are the uprights, L-Py UFO "branch"),
       sidebranch -> shoot (L-Py UFO "tertiarybranch" laterals), spur -> spur

The resize factor is chosen on MFO val under the requested mapping (highest 3-class mIoU,
ties to the smaller factor), then train and test are reported at every factor.

python scripts/rescore_mfo_mapping.py RESULTS_DIR --mapping ufo --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_real_mfo as M  # noqa: E402

from spur_depth.branches.tree import BRANCH, SHOOT, SPUR, TRUNK  # noqa: E402

MAPPINGS = {
    "v2": dict(M.SCORED),
    "ufo": {
        "leader": (M.ROW["leader"], (TRUNK, BRANCH)),
        "sidebranch": (M.ROW["sidebranch"], (SHOOT,)),
        "spur": (M.ROW["spur"], (SPUR,)),
    },
}


def split_metrics(rows: list[dict], mapping: str) -> dict:
    saved = M.SCORED
    M.SCORED = MAPPINGS[mapping]
    try:
        return M.split_metrics(rows)
    finally:
        M.SCORED = saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="directory with mfo_frames_{val,test,train}.json")
    ap.add_argument("--mapping", choices=tuple(MAPPINGS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"mapping": args.mapping, "results_dir": str(args.results), "splits": {}, "inputs": {}}
    for split in ("val", "test", "train"):
        p = args.results / f"mfo_frames_{split}.json"
        if not p.is_file():
            continue
        rows = json.loads(p.read_text())["frames"]
        out["inputs"][split] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "frames": len(rows),
        }
        out["splits"][split] = split_metrics(rows, args.mapping)
    val = out["splits"]["val"]
    best = max(M.FACTORS, key=lambda f: (val[M.fkey(f)]["miou3"] or -1.0, -f))
    out["chosen_factor"] = best
    out["selection_rule"] = (
        "highest MFO-val 3-class mIoU under this mapping; ties -> smaller factor"
    )
    out["headline"] = {
        s: {
            "miou3": m[M.fkey(best)]["miou3"],
            "iou": {k: m[M.fkey(best)][k]["iou"] for k in MAPPINGS[args.mapping]},
            "wood_iou_t": m[M.fkey(best)]["wood_vs_t"]["iou"],
            "labelled_wood_recall": m[M.fkey(best)]["labelled_wood_recall"],
            "background_predicted_wood_rate": m[M.fkey(best)]["background_predicted_wood_rate"],
        }
        for s, m in out["splits"].items()
    }
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"chosen_factor": best, "headline": out["headline"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

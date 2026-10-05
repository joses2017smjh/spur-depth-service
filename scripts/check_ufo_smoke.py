"""Protocol-v3 smoke gate on lpy_ufo_00000 (a train tree): frames, GT surface residual, framing.

python smoke_gate.py OUT_JSON SHEET_PNG
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from spur_depth.branches import gt as G
from spur_depth.branches.tree import CLASSES, IGNORE
from spur_depth.branches.viz import HEX

TREE = "lpy_ufo_00000"
out_json, sheet_png = Path(sys.argv[1]), Path(sys.argv[2])
refs = [G.FrameRef(TREE, r, s, d) for r in G.RIGS for s in G.SHOTS for d in G.SIDES]
present = [r for r in refs if r.path("ann").is_file() and r.path("depth").is_file()]
da2 = [r for r in refs if r.path("Da2Finetune").is_file()]
rows, tiles, model = [], [], None
for ref in present:
    rgb, depth, mask, ann, K, T, fg = G.frame_gt(ref, model=model)
    model = model or G.load_tree_model(TREE, ann)
    lab = (fg.cls > 0) & (fg.cls != IGNORE)
    tree_px = int(mask.sum())
    counts = {CLASSES[c]: int((fg.cls == c).sum()) for c in range(1, len(CLASSES))}
    rows.append(
        {
            "key": ref.key,
            "residual_mm": round(1000 * fg.surface_residual_m, 3),
            "labelled_px": int(lab.sum()),
            "tree_px": tree_px,
            "unmatched_frac": round(float(((fg.cls == IGNORE) & mask).sum()) / max(tree_px, 1), 4),
            "class_px": counts,
            "parts": len(fg.graph.parts),
        }
    )
    if ref.side == "l" and ref.shot in ("shot01", "shot03", "shot06"):
        ov = (rgb * 0.5).astype(np.uint8)
        for c, h in HEX.items():
            col = tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            ov[fg.cls == c] = col
        tile = cv2.resize(np.concatenate([rgb, ov], 1), None, fx=0.25, fy=0.25)
        cv2.putText(tile, ref.key[-28:], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    print(rows[-1]["key"], rows[-1]["residual_mm"], rows[-1]["labelled_px"], flush=True)

res = np.array([r["residual_mm"] for r in rows])
framed = sum(r["labelled_px"] >= 5000 for r in rows)
gate = {
    "frames": len(present),
    "da2_maps": len(da2),
    "median_residual_mm": float(np.nanmedian(res)) if len(res) else None,
    "frames_with_5000_labelled_px": framed,
}
gate["pass"] = bool(
    gate["frames"] >= 60
    and gate["da2_maps"] >= 60
    and gate["median_residual_mm"] is not None
    and gate["median_residual_mm"] <= 2.0
    and framed >= 50
)
class_tot = {c: sum(r["class_px"][c] for r in rows) for c in CLASSES[1:]}
out_json.write_text(
    json.dumps({"tree": TREE, "gate": gate, "class_px_total": class_tot, "frames": rows}, indent=1)
    + "\n"
)
if tiles:
    w = max(t.shape[1] for t in tiles)
    cv2.imwrite(
        str(sheet_png), cv2.cvtColor(np.concatenate([t for t in tiles], 0), cv2.COLOR_RGB2BGR)
    )
print(json.dumps(gate), class_tot)

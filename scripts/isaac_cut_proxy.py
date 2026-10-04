"""Offline CPU proxy: would the Isaac pruner have cut the true branch at the perceived target?

For every observable ground-truth cut of a frame, the predicted graph is built exactly as
``scripts/eval_branches.py`` builds it (its own helpers, the same ``--cfg`` overrides into
``AssembleConfig``). Predicted and true cut targets (``Part.cut()`` on both sides) are paired
with the scorer's own Jain-tier assignment (``metrics._cut_counts``: 5 cm and 30 deg,
one-to-one, maximising successes; the pair count is checked against the scorer's on every
frame). The pruner is then posed at the perceived target with the registered rule below, and
the pruning repo's own geometry, imported from a read-only clone at a fixed commit, decides
whether the jaws would have closed on the true branch:

  legacy  ``ur5e_pruner.yaml`` mouth and failure boxes: ``evaluate_cut_success`` (15 deg),
          ``nearby_wood_in_failure_zone`` over every other L-Py cylinder of the tree
  demo    the vision demo's mouth 70 mm ahead of the tool: ``evaluate_cut_gate`` with the
          demo's CutConfig (8 mm mouth tolerance, 15 deg, radius <= 12 mm)

Pose rule (registered): approach = camera-to-perceived-target ray made perpendicular to the
perceived axis; closing axis = cross(approach, perceived axis) (the vision demo's
``cross(tool_z, branch_axis)``); tool x = closing axis, tool z = approach. The perceived cut
point goes to the centre of the legacy box's success depth, or to the demo mouth.

Truth: the paired ground-truth part's axis between arc lengths [s_c - L/2, s_c + L/2] clipped
to the part (s_c is ``Part.cut()``'s 1.5 cm), with its radius; every other cylinder of the
L-Py tree (occluded ones too) is other wood. Everything is in the camera frame.

Conditions per (method, depth source, jaw model):
  control       perceived = the ground-truth cut itself (harness ceiling)
  open_loop     perceived = the paired predicted cut
  sensed_depth  open_loop minus the error component along the approach (an insertion stop)
Unpaired observable ground-truth cuts count as failures, as in cut recall.

This is a CPU geometry proxy, never an Isaac result: no motion, tracking, contact or physics.

python scripts/isaac_cut_proxy.py --split val --calibrate-length   # truth length, control only
python scripts/isaac_cut_proxy.py --split val --every 5 \
    --pred-dir /nfs/hpc/share/sanchej7/spur-branch-runs/21523238/pred \
    --cfg '{"orphan_cost": 1.2, "junction_bonus": 0.0, "depth_percentile": 50.0}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import eval_branches as EB  # noqa: E402

from spur_depth.branches import gt as G  # noqa: E402
from spur_depth.branches import metrics as M  # noqa: E402
from spur_depth.branches.assemble import AssembleConfig, assemble  # noqa: E402
from spur_depth.branches.dataset import frame_depth, load_cached_frame  # noqa: E402
from spur_depth.branches.depth_fusion import fuse_sensor_mono  # noqa: E402
from spur_depth.branches.tree import (  # noqa: E402
    CLASSES,
    CUT_OFFSET_M,
    IGNORE,
    TRUNK,
    point_at_arclength,  # noqa: E402
)

ISAAC_SHA = "7498eee1d2f28bfe1e7e84bbfe7610501ad19985"
ISAAC_SRC = Path(f"/nfs/hpc/share/sanchej7/spur-branch-eval/isaac-src-{ISAAC_SHA[:12]}")
OUT = Path("/nfs/hpc/share/sanchej7/spur-branch-eval/isaac_proxy")
SELF = "scripts/isaac_cut_proxy.py"
METHODS = ("oracle", "branchnet")
GRAPH_VERSION = 1  # bump if build_graph changes; graphs are cached under the eval fingerprint

# Truth-segment length, registered from the perfect-perception control on val
# (``--calibrate-length``; the rule and the curve are in length_calibration.json).
TRUTH_SEGMENT_M: float | None = 0.10  # registered 2026-10-03 11:47:48 PDT, before any perceived run
CALIBRATION_SHA256 = "cb52eecd8608647702715e7ac12064c3659299b6d5058b09886dff253bc21555"
LENGTH_GRID_M = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
CALIBRATION_MIN = 0.95

DEMO_MOUTH_OFFSET_M = (0.0, 0.0, 0.070)  # vision_demo_controller.py: self.mouth_offset
DEMO_GATE = {"max_branch_radius_m": 0.012, "mouth_position_tolerance_m": 0.008}
TARGET_ID = "proxy_target"
PREFILTER_M = 0.25  # cylinders farther than this from the posed target cannot reach a box
DEGENERATE_SIN = 1e-3  # perceived axis within ~0.06 deg of the viewing ray
# Source lines the registered constants were read from (checked on every run).
SOURCE_CHECKS = {
    "source/isaaclab_pruning/isaaclab_pruning/sim/vision_demo_controller.py": (
        "self.mouth_offset = (0.0, 0.0, 0.070)",
        "max_branch_radius_m=0.012",
        "mouth_position_tolerance_m=0.008",
        "closing_w = np.cross(rotation[:, 2], np.asarray(branch_axis_w, dtype=float))",
    ),
    "hpc/inner/render_pruning_workflow.py": (
        "proxy_closing_w = np.cross(tool_rotation_w[:, 2], "
        "np.asarray(env.blender_scene.target_axis_w))",
    ),
    "source/isaaclab_pruning/isaaclab_pruning/sim/pruning_env.py": (
        "closing_axis = eef_pose.new_tensor([1.0, 0.0, 0.0]).expand(self.num_envs, 3)",
    ),
}
CONDITIONS = ("open_loop", "sensed_depth")
LEGACY_COLS = (
    "jaw_hit",
    "perpendicular",
    "clear_target",
    "clear_other",
    "success_isolated",
    "success",
    "clear_other_surface",
    "success_surface",
)
DEMO_COLS = (
    "jaw_hit",
    "perpendicular",
    "radius_ok",
    "clear_other",
    "gate_only",
    "success",
    "clear_other_surface",
    "success_surface",
)
L_COLS = {
    "legacy": ("jaw_hit", "clear_target", "success_isolated", "success"),
    "demo": ("jaw_hit", "perpendicular", "gate_only", "success"),
}


def _sha(path: Path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(repo: Path, *a: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *a], capture_output=True, text=True
    ).stdout.strip()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------- graphs


def build_graph(method, source, ref, frame, gt_cls, K, cfg, pred_dir, get_depth):
    """The predicted graph exactly as eval_branches.main builds it."""
    net_source = "sensor" if source == "fused" else source
    cls, conf, tweak, _ = EB.predicted_classes(
        method, ref, net_source, gt_cls, frame, pred_dir, None
    )
    if source == "fused":
        wood = (cls > 0) & (cls != IGNORE) if method != "oracle" else gt_cls > 0
        depth, _ = fuse_sensor_mono(get_depth("sensor"), get_depth("da2"), wood)
    else:
        depth = get_depth(source)
    return assemble(cls, depth, K, conf=conf, cfg=replace(cfg, **tweak))


# Everything a cached graph depends on besides the config, method, depth source and inputs:
# the modules on the build path and the source of the two functions that call them. A narrower
# key than eval_branches' fingerprint (all scorer files and the protocol document), so protocol
# and scorer edits elsewhere do not invalidate graphs whose build path is unchanged.
GRAPH_FILES = tuple(
    f"spur_depth/branches/{m}.py"
    for m in ("assemble", "skeleton", "tree", "depth_fusion", "depth_noise", "dataset", "gt")
)


def graph_key(args, cfg, method: str, source: str) -> str:
    import inspect

    blob = {
        "files": {f: _sha(REPO / f) for f in GRAPH_FILES},
        "predicted_classes": hashlib.sha256(
            inspect.getsource(EB.predicted_classes).encode()
        ).hexdigest(),
        "build_graph": hashlib.sha256(inspect.getsource(build_graph).encode()).hexdigest(),
        "assemble_config": asdict(cfg),
        "pred": EB.provenance(args, cfg)["pred"] if method == "branchnet" else None,
        "cache": str(args.cache),
        "method": method,
        "source": source,
        "graph_version": GRAPH_VERSION,
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


class GraphCache:
    """Predicted graphs pickled per (method, depth, frame) under ``graph_key``."""

    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.keys = {(m, s): graph_key(args, cfg, m, s) for m in args.methods for s in args.depths}
        self.dir = args.out / "graphs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.built = self.loaded = 0

    def path(self, method, source, ref) -> Path:
        return self.dir / f"{method}__{source}__{ref.key.replace('/', '__')}.pkl"

    def get(self, method, source, ref, load):
        p = self.path(method, source, ref)
        key = self.keys[(method, source)]
        if p.is_file():
            try:
                d = pickle.loads(p.read_bytes())
                if d.get("graph_key") == key and d["key"] == ref.key:
                    self.loaded += 1
                    return d["graph"]
            except Exception:  # a torn or foreign file is rebuilt
                pass
        gt_cls, K, frame, get_depth = load()
        g = build_graph(
            method, source, ref, frame, gt_cls, K, self.cfg, self.args.pred_dir, get_depth
        )
        blob = {"graph_key": key, "graph_version": GRAPH_VERSION, "key": ref.key, "graph": g}
        _write_atomic(p, pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))
        self.built += 1
        return g


def frame_loader(ref, cache: Path, gt_cls, K):
    state = {}

    def load():
        if "frame" not in state:
            state["frame"] = load_cached_frame(ref, cache, skel=False)
            state["depths"] = {}

        def get_depth(src):
            if src not in state["depths"]:
                state["depths"][src] = frame_depth(src, state["frame"], ref)
            return state["depths"][src]

        return gt_cls, K, state["frame"], get_depth

    return load


# ----------------------------------------------------------------------------- pairing


def gt_cut_list(gtg):
    """GT cuts in metrics._cut_counts order: (pid, point, axis, class, observable)."""
    scored = M.scored_parts(gtg)
    obs = M.observable_edges(gtg, scored)
    out = []
    for p in gtg.parts.values():
        if p.cls == TRUNK or p.parent is None or p.parent not in gtg.parts or p.pid not in scored:
            continue
        pt, ax = p.cut()
        out.append((p.pid, pt, ax, p.cls, (p.pid, p.parent) in obs))
    return out, obs, scored


def jain_pairs(gt_cuts, pred):
    """The scorer's Jain-tier one-to-one assignment, replicated input for input.

    Returns ({gt index: (pred pid, point, axis, dist m, angle deg)}, n don't-care, n pred cuts).
    """
    plist = [
        p for p in pred.parts.values() if p.cls != TRUNK and p.parent is not None and p.length > 0
    ]
    pred_cuts = [p.cut() for p in plist]
    pairs, dont_care = {}, 0
    if gt_cuts and pred_cuts:
        Gp = np.array([g[1] for g in gt_cuts])
        Pp = np.array([p[0] for p in pred_cuts])
        D = np.linalg.norm(Gp[:, None, :] - Pp[None, :, :], axis=-1)
        A = np.array([[M._axis_angle_deg(g[2], p[1]) for p in pred_cuts] for g in gt_cuts])
        dmax, amax = M.CUT_TIERS["jain"]
        S = (D <= dmax) & (A <= amax)
        cost = np.where(S, 1e-3 * D / dmax, 1.0)
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if not S[i, j]:
                continue
            if gt_cuts[i][4]:
                pt, ax = pred_cuts[j]
                pairs[i] = (plist[j].pid, pt, ax, float(D[i, j]), float(A[i, j]))
            else:
                dont_care += 1
    return pairs, dont_care, len(pred_cuts)


# ----------------------------------------------------------------------------- truth


class Wood:
    """Every cylinder of the L-Py tree in the frame's camera, from spur_depth's own GT model."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.models: dict = {}
        self.max_graph_err_m = 0.0

    def frame(self, ref, gtg) -> dict:
        if ref.tree not in self.models:
            ann = json.loads(ref.path("ann").read_text())
            self.models[ref.tree] = G.load_tree_model(ref.tree, ann)
        m = self.models[ref.tree]
        meta = json.loads((self.cache / ref.tree / ref.rig / f"{ref.stem}.meta.json").read_text())
        T = np.asarray(meta["T_wc"], dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        for p in gtg.parts.values():  # the cached GT graph must be this model in this camera
            q = m.part_points[p.pid] @ R.T + t
            err = max(np.linalg.norm(q[0] - p.points[0]), np.linalg.norm(q[-1] - p.points[-1]))
            if err > 1e-3:
                raise RuntimeError(f"{ref.key}: GT part {p.pid} is {err:.4f} m off the tree model")
            self.max_graph_err_m = max(self.max_graph_err_m, float(err))
        return {"a": m.seg_a @ R.T + t, "b": m.seg_b @ R.T + t, "r": m.seg_r, "part": m.seg_part}

    def relation(self, tree: str, q: int, target: int) -> str:
        m = self.models[tree]
        par = m.part_parent
        if q == par[target]:
            rel = "parent"
        elif par[q] == target:
            rel = "child"
        elif par[q] >= 0 and par[q] == par[target]:
            rel = "sibling"
        elif par[target] >= 0 and q == par[par[target]]:
            rel = "grandparent"
        else:
            rel = "other"
        return f"{rel}:{CLASSES[int(m.part_cls[q])]}"


def truth_segment(part, length_m: float):
    """The true axis between arc lengths s_c -/+ L/2 (s_c as in Part.cut), clipped to the part."""
    s_c = min(CUT_OFFSET_M, 0.5 * part.length)
    s0, s1 = max(0.0, s_c - 0.5 * length_m), min(part.length, s_c + 0.5 * length_m)
    return point_at_arclength(part.points, s0)[0], point_at_arclength(part.points, s1)[0]


def seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n, K) distances from points to segments [A_k, B_k]."""
    d = B - A
    dd = np.maximum((d * d).sum(1), 1e-18)
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(-1) / dd[None], 0.0, 1.0)
    return np.linalg.norm(A[None] + t[..., None] * d[None] - P[:, None, :], axis=-1)


# ----------------------------------------------------------------------------- pose


def tool_frame(point: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, bool]:
    """Registered pose rule: (tool-to-camera rotation, degenerate flag).

    Columns: x = closing axis = cross(approach, axis), y = z cross x (= -axis), z = approach =
    camera-to-target ray made perpendicular to the perceived axis. If the axis lies along the
    ray (within ~0.06 deg) the camera's optical axis, then its y axis, stands in for the ray.
    """
    u = axis / np.linalg.norm(axis)
    ray = point / np.linalg.norm(point)
    a = ray - (ray @ u) * u
    degenerate = bool(np.linalg.norm(a) < DEGENERATE_SIN)
    if degenerate:
        for f in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            a = f - (f @ u) * u
            if np.linalg.norm(a) > 0.1:
                break
    a = a / np.linalg.norm(a)
    c = np.cross(a, u)
    c = c / np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a]), degenerate


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


# ----------------------------------------------------------------------------- Isaac geometry


class Isaac:
    """The pruning repo's geometry, imported from the read-only clone (never the live tree)."""

    def __init__(self, src: Path):
        src = Path(src)
        pkg = src / "source" / "isaaclab_pruning"
        if not (pkg / "isaaclab_pruning" / "__init__.py").is_file():
            raise FileNotFoundError(f"no pruning clone at {src}")
        sys.path.insert(0, str(pkg))
        import isaaclab_pruning
        import torch
        from isaaclab_pruning.geometry.cutter import cutter_boxes_from_spec
        from isaaclab_pruning.geometry.wood import nearby_wood_in_failure_zone
        from isaaclab_pruning.perception.jaw_self_mask import (
            JAW_HALF_EXTENTS_M,
            JAW_HEIGHT_M,
            JAW_OPEN_GAP_M,
            proxy_roll_rad,
        )
        from isaaclab_pruning.robot import load_ur5e_pruner_spec
        from isaaclab_pruning.sim.vision_demo_controller import closing_axis_tool_at
        from isaaclab_pruning.task.simulated_cut import (
            CutConfig,
            CutObservation,
            evaluate_cut_gate,
            tool_mouth_geometry,
        )
        from isaaclab_pruning.task.success import (
            OrientedBox,
            evaluate_cut_success,
            segment_intersects_obb,
        )

        if not Path(isaaclab_pruning.__file__).resolve().is_relative_to(src.resolve()):
            raise RuntimeError(f"isaaclab_pruning imported from {isaaclab_pruning.__file__}")
        self.src, self.torch = src, torch
        self.cutter_boxes_from_spec = cutter_boxes_from_spec
        self.nearby_wood_in_failure_zone = nearby_wood_in_failure_zone
        self.closing_axis_tool_at = closing_axis_tool_at
        self.CutObservation, self.evaluate_cut_gate = CutObservation, evaluate_cut_gate
        self.tool_mouth_geometry = tool_mouth_geometry
        self.OrientedBox = OrientedBox
        self.evaluate_cut_success = evaluate_cut_success
        self.segment_intersects_obb = segment_intersects_obb
        sp = self.spec = load_ur5e_pruner_spec()
        # Legacy success depth: above the failure box, inside the mouth box (tool z).
        z_lo = max(
            sp.failure_offset_m[2] + sp.failure_half_extents_m[2],
            sp.mouth_offset_m[2] - sp.mouth_half_extents_m[2],
        )
        z_hi = sp.mouth_offset_m[2] + sp.mouth_half_extents_m[2]
        self.legacy_window_z_m = (z_lo, z_hi)
        self.legacy_target_tool = np.array(
            [sp.mouth_offset_m[0], sp.mouth_offset_m[1], 0.5 * (z_lo + z_hi)]
        )
        self.gate_cfg = CutConfig(selected_target_id=TARGET_ID, **DEMO_GATE)
        # Open-jaw sweep of the demo's visual jaws (both jaws and the gap between them).
        self.sweep_half = np.array(
            [
                JAW_OPEN_GAP_M / 2 + 2 * JAW_HALF_EXTENTS_M[0],
                JAW_HALF_EXTENTS_M[1],
                JAW_HALF_EXTENTS_M[2],
            ]
        )
        self.sweep_centre_tool = np.array([0.0, 0.0, JAW_HEIGHT_M])
        if abs(proxy_roll_rad((1.0, 0.0, 0.0))) > 1e-12 or JAW_HEIGHT_M != DEMO_MOUTH_OFFSET_M[2]:
            raise RuntimeError("demo jaw surrogate no longer centred on the mouth along tool x")
        self.jaw = {
            "half_extents_m": list(JAW_HALF_EXTENTS_M),
            "open_gap_m": JAW_OPEN_GAP_M,
            "height_m": JAW_HEIGHT_M,
        }

    def t64(self, x):
        return self.torch.as_tensor(np.ascontiguousarray(x), dtype=self.torch.float64)

    def legacy_boxes(self, poses: np.ndarray):
        sp = self.spec
        return self.cutter_boxes_from_spec(
            eef_pose_w=self.t64(poses),
            mouth_half_extents=sp.mouth_half_extents_m,
            failure_half_extents=sp.failure_half_extents_m,
            failure_offset_eef=sp.failure_offset_m,
            mouth_offset_eef=sp.mouth_offset_m,
        )

    def segment_hits(self, starts, ends, box, inflate=None) -> np.ndarray:
        """(n, m) per-segment box tests; ``inflate`` (n, m) grows the box by each wood radius."""
        n, m = starts.shape[:2]
        half = box.half_extents.expand(n, 3).repeat_interleave(m, 0)
        if inflate is not None:
            half = half + self.t64(inflate.reshape(-1, 1))
        flat = self.OrientedBox(
            center_w=box.center_w.repeat_interleave(m, 0),
            rotation_bw=box.rotation_bw.repeat_interleave(m, 0),
            half_extents=half,
        )
        hit = self.segment_intersects_obb(
            self.t64(starts.reshape(-1, 3)), self.t64(ends.reshape(-1, 3)), flat
        )
        return hit.numpy().reshape(n, m)

    def source_checks(self) -> dict:
        out = {}
        for rel, snippets in SOURCE_CHECKS.items():
            text = (self.src / rel).read_text()
            missing = [s for s in snippets if s not in text]
            if missing:
                raise RuntimeError(f"{rel} no longer contains {missing}")
            out[rel] = {"sha256": _sha(self.src / rel), "lines": list(snippets)}
        return out


def evaluate_items(I: Isaac, items: list[dict], wood: dict, lengths) -> list[dict]:
    """Score every posed item at every truth length, batched per frame."""
    n, nl = len(items), len(lengths)
    if n == 0:
        return []
    P = np.array([it["p"] for it in items])
    Rs = np.array([it["R"] for it in items])
    quat = np.array([quat_wxyz(R) for R in Rs])
    tgt = np.array([it["tgt"] for it in items])

    # Other wood: every cylinder of the tree except the target part's own, near the target.
    A, B, rad, part = wood["a"], wood["b"], wood["r"], wood["part"]
    near = (seg_dist(P, A, B) <= PREFILTER_M) & (part[None, :] != tgt[:, None])
    m = max(int(near.sum(1).max()), 1)
    idx = np.full((n, m), -1)
    for k in range(n):
        j = np.nonzero(near[k])[0]
        idx[k, : len(j)] = j
    pad = idx < 0
    safe = np.where(pad, 0, idx)
    oa, ob = A[safe], B[safe]
    v = ob - oa
    ln = np.linalg.norm(v, axis=-1)
    cen = np.where(pad[..., None], 1e3, 0.5 * (oa + ob))
    ax = np.where(pad[..., None], np.array([1.0, 0.0, 0.0]), v / np.maximum(ln, 1e-12)[..., None])
    ln = np.where(pad, 1e-3, ln)
    orad = np.where(pad, 0.0, rad[safe])
    starts, ends = cen - 0.5 * ln[..., None] * ax, cen + 0.5 * ln[..., None] * ax

    # Legacy: mouth and failure boxes at the registered placement.
    leg_pose = np.hstack([P - Rs @ I.legacy_target_tool, quat])
    mouth, failure = I.legacy_boxes(leg_pose)
    if float((mouth.rotation_bw.numpy() - Rs).__abs__().max()) > 1e-9:
        raise RuntimeError("quaternion round trip changed the tool frame")
    other = I.nearby_wood_in_failure_zone(
        I.t64(cen), I.t64(ax), I.t64(ln), failure, exclude_mask=I.t64(pad).bool()
    ).numpy()
    leg_hit = I.segment_hits(starts, ends, failure) & ~pad
    if not np.array_equal(leg_hit.any(1), other):
        raise RuntimeError("per-segment hits disagree with nearby_wood_in_failure_zone")
    leg_hit_s = I.segment_hits(starts, ends, failure, inflate=orad) & ~pad

    # Demo: open-jaw sweep around the mouth (the perceived point).
    demo_origin = P - Rs @ np.asarray(DEMO_MOUTH_OFFSET_M)
    sweep = I.OrientedBox(
        center_w=I.t64(demo_origin + Rs @ I.sweep_centre_tool),
        rotation_bw=I.t64(Rs),
        half_extents=I.t64(np.tile(I.sweep_half, (n, 1))),
    )
    demo_hit = I.segment_hits(starts, ends, sweep) & ~pad
    demo_hit_s = I.segment_hits(starts, ends, sweep, inflate=orad) & ~pad

    # True target segments per length.
    SA = np.zeros((n, nl, 3))
    SB = np.zeros((n, nl, 3))
    seg_cache: dict = {}
    for k, it in enumerate(items):
        for li, L in enumerate(lengths):
            key = (it["tgt"], L)
            if key not in seg_cache:
                seg_cache[key] = truth_segment(it["gt"], L)
            SA[k, li], SB[k, li] = seg_cache[key]
    sa, sb = SA.reshape(-1, 3), SB.reshape(-1, 3)
    sv = sb - sa
    sl = np.linalg.norm(sv, axis=1)
    sax = sv / sl[:, None]
    mouth_r, failure_r = I.legacy_boxes(np.repeat(leg_pose, nl, axis=0))
    common = dict(
        branch_centroid_w=I.t64(0.5 * (sa + sb)),
        branch_axis_w=I.t64(sax),
        branch_length_m=I.t64(sl),
        cutter_closing_axis_w=mouth_r.rotation_bw[:, :, 0],
        mouth_box=mouth_r,
        failure_box=failure_r,
        perpendicularity_tolerance_deg=I.spec.perpendicularity_tolerance_deg,
    )
    iso = I.evaluate_cut_success(**common)
    full = I.evaluate_cut_success(
        **common, other_wood_in_failure_zone=I.torch.as_tensor(np.repeat(other, nl))
    )

    def nl_(t):
        return t.numpy().reshape(n, nl)

    surf_clear = ~leg_hit_s.any(1)
    out = []
    for k, it in enumerate(items):
        legacy = {
            "jaw_hit": nl_(iso.mouth_hit)[k],
            "perpendicular": nl_(iso.perpendicular)[k],
            "clear_target": nl_(iso.failure_clear)[k],
            "clear_other": np.full(nl, not other[k]),
            "success_isolated": nl_(iso.success)[k],
            "success": nl_(full.success)[k],
            "clear_other_surface": np.full(nl, surf_clear[k]),
            "success_surface": nl_(iso.success)[k] & surf_clear[k],
        }
        pose = tuple(float(x) for x in np.r_[demo_origin[k], quat[k]])
        closing_tool = I.closing_axis_tool_at(tuple(quat[k]), tuple(it["u"]))
        if not np.allclose(closing_tool, (1.0, 0.0, 0.0), atol=1e-9):
            raise RuntimeError(f"closing_axis_tool_at gave {closing_tool}, not tool x")
        mouth_w, closing_w = I.tool_mouth_geometry(pose, DEMO_MOUTH_OFFSET_M, closing_tool)
        if not np.allclose(mouth_w, P[k], atol=1e-9):
            raise RuntimeError("demo mouth is not at the perceived point")
        hazard = bool(demo_hit[k].any())
        clear_s = not bool(demo_hit_s[k].any())
        demo = {c: np.zeros(nl, dtype=bool) for c in DEMO_COLS}
        mouth_mm, perp_deg = np.zeros(nl), np.zeros(nl)
        mw = np.asarray(mouth_w)
        for li in range(nl):
            a, b = SA[k, li], SB[k, li]
            d = b - a
            t = float(np.clip((mw - a) @ d / max(d @ d, 1e-18), 0.0, 1.0))
            obs = I.CutObservation(
                time_s=0.0,
                vision_timestamp_s=0.0,
                target_id=TARGET_ID,
                target_semantic="branch",  # as VisionPruningDemo passes it
                vision_valid=True,
                target_position_w=tuple(float(x) for x in a + t * d),
                target_axis_w=tuple(float(x) for x in d / np.linalg.norm(d)),
                target_radius_m=float(it["radius"]),
                mouth_position_w=mouth_w,
                cutter_closing_axis_w=closing_w,
                hazard_contact=False,
            )
            c0 = I.evaluate_cut_gate(obs, I.gate_cfg)
            c1 = I.evaluate_cut_gate(replace(obs, hazard_contact=hazard), I.gate_cfg)
            demo["jaw_hit"][li] = "outside_mouth_tolerance" not in c0.reasons
            demo["perpendicular"][li] = "closing_axis_not_perpendicular" not in c0.reasons
            demo["radius_ok"][li] = not (
                {"invalid_branch_radius", "branch_exceeds_demo_cut_limit"} & set(c0.reasons)
            )
            demo["clear_other"][li] = not hazard
            demo["gate_only"][li] = c0.ready_to_close
            demo["success"][li] = c1.ready_to_close
            demo["clear_other_surface"][li] = clear_s
            demo["success_surface"][li] = c0.ready_to_close and clear_s
            mouth_mm[li] = 1e3 * c0.mouth_distance_m
            perp_deg[li] = c0.perpendicularity_error_deg
        out.append(
            {
                "legacy": legacy,
                "demo": demo,
                "legacy_perp_deg": nl_(iso.perpendicularity_error_deg)[k],
                "demo_mouth_mm": mouth_mm,
                "demo_perp_deg": perp_deg,
                "legacy_hits": sorted({int(wood["part"][j]) for j in idx[k][leg_hit[k]]}),
                "demo_hits": sorted({int(wood["part"][j]) for j in idx[k][demo_hit[k]]}),
                "legacy_hits_surface": sorted({int(wood["part"][j]) for j in idx[k][leg_hit_s[k]]}),
                "demo_hits_surface": sorted({int(wood["part"][j]) for j in idx[k][demo_hit_s[k]]}),
            }
        )
    return out


# ----------------------------------------------------------------------------- frames

ROW_COLS = (
    "key",
    "tree",
    "method",
    "depth",
    "condition",
    "gt_pid",
    "gt_class",
    "paired",
    "pred_pid",
    "d_mm",
    "ang_deg",
    "e_app_mm",
    "e_close_mm",
    "degenerate",
    *(f"legacy_{c}" for c in LEGACY_COLS),
    "legacy_perp_deg",
    "legacy_hits",
    "legacy_hits_surface",
    *(f"demo_{c}" for c in DEMO_COLS),
    "demo_mouth_mm",
    "demo_perp_deg",
    "demo_hits",
    "demo_hits_surface",
)


def make_item(point, axis, gt_part) -> dict:
    point, axis = np.asarray(point, dtype=np.float64), np.asarray(axis, dtype=np.float64)
    R, degenerate = tool_frame(point, axis)
    return {
        "p": point,
        "u": axis / np.linalg.norm(axis),
        "R": R,
        "degenerate": degenerate,
        "gt": gt_part,
        "tgt": gt_part.pid,
        "radius": float(np.median(gt_part.radius)),
    }


def score_frame(ref, args, I, cache, wood: Wood, lengths, control_only=False):
    """Rows (one per observable GT cut x method/depth x condition) and scorer-parity records."""
    gtg, gt_cls, K = EB.gt_graph(ref, args.cache)
    w = wood.frame(ref, gtg)
    gt_cuts, obs, scored = gt_cut_list(gtg)
    obs_idx = [i for i, g in enumerate(gt_cuts) if g[4]]
    items, rows, parity = [], [], []

    def base_row(method, depth, cond, i):
        r = dict.fromkeys(ROW_COLS)
        r.update(key=ref.key, tree=ref.tree, method=method, depth=depth, condition=cond)
        r.update(gt_pid=int(gt_cuts[i][0]), gt_class=CLASSES[gt_cuts[i][3]], paired=0)
        r.update(pred_pid=-1, degenerate=0)
        return r

    for i in obs_idx:
        pid, pt, ax = gt_cuts[i][:3]
        it = make_item(pt, ax, gtg.parts[pid])
        row = base_row("truth", "-", "control", i)
        row.update(paired=1, d_mm=0.0, ang_deg=0.0, e_app_mm=0.0, e_close_mm=0.0)
        row["degenerate"] = int(it["degenerate"])
        it["row"] = row
        items.append(it)
        rows.append(row)
    if not control_only:
        load = frame_loader(ref, args.cache, gt_cls, K)
        for source in args.depths:
            for method in args.methods:
                pred = cache.get(method, source, ref, load)
                pairs, dont_care, n_pred = jain_pairs(gt_cuts, pred)
                cc = M._cut_counts(gtg, pred, obs, scored)
                rec = {
                    "key": ref.key,
                    "method": method,
                    "depth": source,
                    "pairs": len(pairs),
                    "scorer_success_jain": cc["success_jain"],
                    "n_gt": len(obs_idx),
                    "scorer_n_gt": cc["n_gt"],
                    "n_pred_jain": n_pred - dont_care,
                    "scorer_n_pred_jain": cc["n_pred_jain"],
                }
                rec["ok"] = (
                    rec["pairs"] == rec["scorer_success_jain"]
                    and rec["n_gt"] == rec["scorer_n_gt"]
                    and rec["n_pred_jain"] == rec["scorer_n_pred_jain"]
                )
                if not rec["ok"]:
                    raise RuntimeError(f"pairing differs from the scorer: {rec}")
                parity.append(rec)
                for i in obs_idx:
                    if i not in pairs:  # no perceived target: a failure, as in cut recall
                        rows += [base_row(method, source, c, i) for c in CONDITIONS]
                        continue
                    ppid, ppt, pax, dist, ang = pairs[i]
                    pid, gpt = gt_cuts[i][0], gt_cuts[i][1]
                    it = make_item(ppt, pax, gtg.parts[pid])
                    e = ppt - gpt
                    app, clo = it["R"][:, 2], it["R"][:, 0]
                    for cond in CONDITIONS:
                        it2 = dict(it)
                        if cond == "sensed_depth":  # insertion stop: no error along the approach
                            it2["p"] = ppt - (e @ app) * app
                        row = base_row(method, source, cond, i)
                        row.update(paired=1, pred_pid=int(ppid), d_mm=1e3 * dist, ang_deg=ang)
                        row["e_app_mm"] = 1e3 * float(e @ app) if cond == "open_loop" else 0.0
                        row["e_close_mm"] = 1e3 * float(e @ clo)
                        row["degenerate"] = int(it["degenerate"])
                        it2["row"] = row
                        items.append(it2)
                        rows.append(row)
    for it, res in zip(items, evaluate_items(I, items, w, lengths)):
        row = it["row"]
        row["_res"] = res
        for k in ("legacy_hits", "demo_hits", "legacy_hits_surface", "demo_hits_surface"):
            row[k] = "|".join(sorted({wood.relation(ref.tree, q, row["gt_pid"]) for q in res[k]}))
    return rows, parity


def fill_registered(rows, li: int) -> None:
    """Row columns at the registered truth length; unpaired rows are failures."""
    for r in rows:
        res = r.pop("_res", None)
        for model, cols in (("legacy", LEGACY_COLS), ("demo", DEMO_COLS)):
            for c in cols:
                r[f"{model}_{c}"] = int(res[model][c][li]) if res else 0
        if res:
            r["legacy_perp_deg"] = round(float(res["legacy_perp_deg"][li]), 2)
            r["demo_mouth_mm"] = round(float(res["demo_mouth_mm"][li]), 2)
            r["demo_perp_deg"] = round(float(res["demo_perp_deg"][li]), 2)
        for k in ("d_mm", "ang_deg", "e_app_mm", "e_close_mm"):
            if r[k] is not None:
                r[k] = round(float(r[k]), 2)


def accumulate(sens: dict, rows, nl: int) -> None:
    """Per-length counts (sensitivity), before ``fill_registered`` drops the arrays."""
    for r in rows:
        g = sens.setdefault(
            f"{r['method']}/{r['depth']}/{r['condition']}",
            {"n": 0, **{m: {c: np.zeros(nl, int) for c in cs} for m, cs in L_COLS.items()}},
        )
        g["n"] += 1
        res = r.get("_res")
        if res:
            for m, cs in L_COLS.items():
                for c in cs:
                    g[m][c] += res[m][c].astype(int)


def _ratio(a, b):
    return None if not b else round(float(a) / float(b), 4)


def summarize(rows) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(f"{r['method']}/{r['depth']}/{r['condition']}", []).append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        n, npair = len(rs), sum(r["paired"] for r in rs)
        g = {"n_gt_cuts": n, "n_paired": npair, "paired_rate": _ratio(npair, n)}
        for model, cols in (("legacy", LEGACY_COLS), ("demo", DEMO_COLS)):
            cnt = {c: int(sum(r[f"{model}_{c}"] for r in rs)) for c in cols}
            g[model] = {
                "rate": {c: _ratio(v, n) for c, v in cnt.items()},
                "rate_given_paired": {c: _ratio(v, npair) for c, v in cnt.items()},
                "counts": cnt,
                "success_per_tree": {
                    t: _ratio(
                        sum(r[f"{model}_success"] for r in rs if r["tree"] == t),
                        sum(1 for r in rs if r["tree"] == t),
                    )
                    for t in sorted({r["tree"] for r in rs})
                },
                "success_per_gt_class": {
                    k: _ratio(
                        sum(r[f"{model}_success"] for r in rs if r["gt_class"] == k),
                        sum(1 for r in rs if r["gt_class"] == k),
                    )
                    for k in sorted({r["gt_class"] for r in rs})
                },
                "other_wood_hits": _hit_counts(rs, f"{model}_hits"),
            }
        g["degenerate_poses"] = int(sum(r["degenerate"] for r in rs))
        out[name] = g
    return out


def _hit_counts(rs, key) -> dict:
    out: dict = {}
    for r in rs:
        for h in (r[key] or "").split("|"):
            if h:
                out[h] = out.get(h, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def sensitivity(sens: dict, lengths) -> dict:
    return {
        name: {
            "n_gt_cuts": g["n"],
            **{
                m: {
                    c: dict(zip(map(str, lengths), (g[m][c] / max(g["n"], 1)).round(4).tolist()))
                    for c in cs
                }
                for m, cs in L_COLS.items()
            },
        }
        for name, g in sorted(sens.items())
    }


def print_table(summary: dict) -> None:
    for model, cols in (("legacy", LEGACY_COLS[:6]), ("demo", DEMO_COLS[:6])):
        print(f"\n{model}: rate over observable GT cuts")
        print(f"{'group':34s} {'n':>5s} {'paired':>7s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
        for name, g in summary.items():
            vals = " ".join(f"{(g[model]['rate'][c] or 0):9.3f}" for c in cols)
            print(f"{name:34s} {g['n_gt_cuts']:5d} {g['paired_rate'] or 0:7.3f} {vals}")


def rows_table(rows) -> dict:
    """Per-pair records as a table; ``frame`` indexes ``frames`` (the tree is in the key)."""
    frames = sorted({r["key"] for r in rows})
    index = {k: i for i, k in enumerate(frames)}
    cols = ["frame", *(c for c in ROW_COLS if c not in ("key", "tree"))]
    body = [[index[r["key"]], *(r[c] for c in cols[1:])] for r in rows]
    return {"frames": frames, "columns": cols, "rows": body}


# ----------------------------------------------------------------------------- provenance


def provenance(args, cfg, I: Isaac) -> dict:
    files = [*EB.SCORER_FILES, SELF]
    status = _git(REPO, "status", "--porcelain", "--untracked-files=all", "--", *files)
    import spur_depth

    sp = I.spec
    return {
        "script_sha256": _sha(REPO / SELF),
        "spur_depth": {
            "commit": _git(REPO, "rev-parse", "HEAD"),
            "files_status": status,
            "dirty": bool(status),
            "spur_depth_module": str(Path(spur_depth.__file__).resolve()),
            "module_under_repo": Path(spur_depth.__file__).resolve().is_relative_to(REPO),
        },
        "eval_provenance": EB.provenance(args, cfg),
        "cfg_overrides": json.loads(args.cfg),
        "assemble_config": asdict(cfg),
        "isaac_clone": {
            "path": str(I.src),
            "registered_sha": ISAAC_SHA,
            "head": _git(I.src, "rev-parse", "HEAD"),
            "tracked_status": _git(I.src, "status", "--porcelain", "--untracked-files=no"),
            "source_checks": I.source_checks(),
        },
        "registered": {
            "pose_rule": "approach = camera-to-perceived-target ray made perpendicular to the "
            "perceived axis (camera at the origin of the OpenCV camera frame); closing axis = "
            "normalize(cross(approach, perceived axis)) as in render_pruning_workflow.py "
            "proxy_closing_w and closing_axis_tool_at; tool x = closing axis, tool z = approach, "
            "tool y = z cross x = -perceived axis (base->tip axis from Part.cut())",
            "closing_axis_sign": "the demo flips its PCA axis so the largest world component is "
            "positive (blender_component.py), which has no camera-frame equivalent; here the "
            "base->tip Part.cut() axis is used. The sign matters only for the legacy boxes' "
            "+y offset (mouth y in [-12.6, 42.0] mm, failure y in [-65.5, 94.5] mm); the demo "
            "jaws are symmetric",
            "degenerate_rule": f"if |ray - (ray.u)u| < {DEGENERATE_SIN}, camera z then y stands in",
            "legacy": {
                "spec_source": sp.cutter_source,
                "mouth_half_extents_m": list(sp.mouth_half_extents_m),
                "mouth_offset_m": list(sp.mouth_offset_m),
                "failure_half_extents_m": list(sp.failure_half_extents_m),
                "failure_offset_m": list(sp.failure_offset_m),
                "perpendicularity_tolerance_deg": sp.perpendicularity_tolerance_deg,
                "success_window_tool_z_m": list(I.legacy_window_z_m),
                "perceived_point_in_tool_m": I.legacy_target_tool.tolist(),
                "closing_axis": "tool x (pruning_env._compute_cut_success)",
                "other_wood": "nearby_wood_in_failure_zone over every other cylinder (axis lines)",
            },
            "demo": {
                "mouth_offset_tool_m": list(DEMO_MOUTH_OFFSET_M),
                "gate": {
                    "max_branch_radius_m": I.gate_cfg.max_branch_radius_m,
                    "mouth_position_tolerance_m": I.gate_cfg.mouth_position_tolerance_m,
                    "alignment_tolerance_deg": I.gate_cfg.alignment_tolerance_deg,
                },
                "gate_inputs": "target = closest point of the true segment to the mouth, axis = "
                "true segment direction, radius = true median radius, semantic 'branch', age 0",
                "jaw_sweep_box": {
                    "centre_tool_m": I.sweep_centre_tool.tolist(),
                    "half_extents_m": I.sweep_half.tolist(),
                    "from": I.jaw,
                    "note": "proxy-specific clearance (other wood inside the open jaws' sweep); "
                    "Isaac's gate sees only contact-sensor hazards and its jaws are visual, so "
                    "'success' is stricter than Isaac's own gate ('gate_only')",
                },
            },
            "truth": {
                "segment": "GT part axis between arc lengths max(0, s_c - L/2) and "
                "min(len, s_c + L/2), s_c = min(0.015, len/2) as Part.cut(); straight chord",
                "truth_segment_m": TRUTH_SEGMENT_M,
                "length_grid_m": list(LENGTH_GRID_M),
                "other_wood": "every other cylinder of the L-Py tree model (occluded too), camera "
                "frame via the cache's T_wc; the target part's own cylinders are excluded",
                "radius": "primary columns use axis lines as the Isaac primitives do; *_surface "
                "columns grow each box by the other wood's radius (target never inflated)",
                "prefilter_m": PREFILTER_M,
            },
            "sensed_depth": "p' = p - ((p - p_gt).a) a with the open-loop approach a and the same "
            "tool rotation (a live insertion stop zeroes the depth error along the approach)",
            "pairing": "metrics._cut_counts Jain tier (5 cm, 30 deg), replicated and checked per "
            "frame; unpaired observable GT cuts are failures; don't-care pairs are skipped",
        },
    }


def guard_test(prov: dict, args) -> None:
    problems = []
    if prov["spur_depth"]["dirty"]:
        problems.append(f"uncommitted scorer files:\n{prov['spur_depth']['files_status']}")
    if not prov["spur_depth"]["module_under_repo"]:
        problems.append(f"spur_depth imported from {prov['spur_depth']['spur_depth_module']}")
    clone = prov["isaac_clone"]
    if clone["head"] != ISAAC_SHA or clone["tracked_status"]:
        problems.append(f"pruning clone not clean at {ISAAC_SHA}")
    cal = args.calibration
    if TRUTH_SEGMENT_M is None or not cal.is_file():
        problems.append("truth length not registered")
    elif json.loads(cal.read_text())["chosen_truth_segment_m"] != TRUTH_SEGMENT_M:
        problems.append("TRUTH_SEGMENT_M differs from length_calibration.json")
    elif _sha(cal) != CALIBRATION_SHA256:
        problems.append(f"{cal} is not the registered calibration file")
    if problems:
        sys.exit("refusing test scoring:\n" + "\n".join(problems))


# ----------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--methods", nargs="+", default=["branchnet", "oracle"], choices=METHODS)
    ap.add_argument("--depths", nargs="+", default=["gt", "fused"], choices=EB.DEPTHS)
    ap.add_argument("--cache", type=Path, default=EB.CACHE)
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="first N frames per tree (val only)")
    ap.add_argument("--every", type=int, default=1, help="every Nth frame per tree (val only)")
    ap.add_argument("--cfg", type=str, default="{}", help="JSON AssembleConfig overrides")
    ap.add_argument("--isaac-src", type=Path, default=ISAAC_SRC)
    ap.add_argument(
        "--calibration",
        type=Path,
        default=OUT / "length_calibration.json",
        help="the val registration of TRUTH_SEGMENT_M (--calibrate-length writes OUT's copy)",
    )
    ap.add_argument("--graphs-only", action="store_true", help="build/cache predicted graphs")
    ap.add_argument("--reverse", action="store_true", help="--graphs-only from the far end")
    ap.add_argument(
        "--calibrate-length", action="store_true", help="control only, over LENGTH_GRID_M (val)"
    )
    args = ap.parse_args(argv)
    if "branchnet" in args.methods and args.pred_dir is None and not args.calibrate_length:
        ap.error("--methods branchnet needs --pred-dir")
    if args.split == "test":
        if args.limit or args.every != 1:
            ap.error("--split test scores every frame: no --limit / --every")
        if args.calibrate_length or args.graphs_only:
            ap.error("--split test runs the registered proxy only")
    return args


def graphs_only(args, cfg) -> int:
    cache = GraphCache(args, cfg)
    frames = EB.frame_list(args)
    if args.every == 1 and not args.limit:  # the --every 5 dev subset first
        first = {r.key for r in EB.frame_list(argparse.Namespace(**{**vars(args), "every": 5}))}
        frames = [r for r in frames if r.key in first] + [r for r in frames if r.key not in first]
    if args.reverse:  # a second builder meets the first in the middle (atomic writes)
        frames = frames[::-1]
    t0 = time.time()
    for n, ref in enumerate(frames):
        gtg, gt_cls, K = EB.gt_graph(ref, args.cache)
        load = frame_loader(ref, args.cache, gt_cls, K)
        for source in args.depths:
            for method in args.methods:
                cache.get(method, source, ref, load)
        print(
            f"[{n + 1}/{len(frames)}] {ref.key} built {cache.built} loaded {cache.loaded} "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    return 0


def _frames_meta(frames) -> dict:
    keys = sorted(r.key for r in frames)
    return {
        "n_frames": len(keys),
        "frame_list_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "trees": sorted({r.tree for r in frames}),
    }


def calibrate(args, I: Isaac, prov: dict) -> int:
    """Perfect-perception control over LENGTH_GRID_M; writes the length registration."""
    if args.split != "val":
        sys.exit("--calibrate-length runs on val only")
    frames = EB.frame_list(args)
    wood, sens, rows = Wood(args.cache), {}, []
    t0 = time.time()
    for n, ref in enumerate(frames):
        fr, _ = score_frame(ref, args, I, None, wood, LENGTH_GRID_M, control_only=True)
        accumulate(sens, fr, len(LENGTH_GRID_M))
        rows += fr
        if (n + 1) % 20 == 0:
            print(f"[{n + 1}/{len(frames)}] {time.time() - t0:.0f}s", flush=True)
    g = sens["truth/-/control"]
    curve = {
        str(L): {m: {c: round(float(g[m][c][li]) / g["n"], 4) for c in L_COLS[m]} for m in L_COLS}
        for li, L in enumerate(LENGTH_GRID_M)
    }
    iso = {
        L: min(curve[str(L)]["legacy"]["success_isolated"], curve[str(L)]["demo"]["gate_only"])
        for L in LENGTH_GRID_M
    }
    best = max(iso.values())
    admissible = [L for L in LENGTH_GRID_M if iso[L] >= min(best, CALIBRATION_MIN)]
    chosen = max(admissible)
    li = LENGTH_GRID_M.index(chosen)
    fill_registered(rows, li)
    summ = summarize(rows)["truth/-/control"]
    iso_fail = [
        {
            k: r[k]
            for k in (
                "key",
                "gt_pid",
                "gt_class",
                "legacy_jaw_hit",
                "legacy_perpendicular",
                "legacy_clear_target",
                "demo_jaw_hit",
                "demo_perpendicular",
                "demo_radius_ok",
                "legacy_perp_deg",
                "demo_mouth_mm",
            )
        }
        for r in rows
        if not (r["legacy_success_isolated"] and r["demo_gate_only"])
    ]
    out = {
        "what": "truth-segment length registration from the perfect-perception control only "
        "(perceived = the GT cut); no predicted graph is built or read",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        **_frames_meta(frames),
        "n_gt_cuts": g["n"],
        "rule": "S(L) = min over jaw models of the control's isolated success (legacy: jaw hit "
        "and perpendicular and target clear of the failure box; demo: evaluate_cut_gate without "
        "hazard). Admissible L: S(L) >= min(max_L S(L), 0.95). Chosen: the largest admissible L. "
        "The grid stops at 0.10 m: the Jain tier tolerates 5 cm of cut-point error, so the true "
        "branch within 5 cm either side of the cut point is the wood a correct cut may land on; "
        "a longer segment would credit cuts outside the protocol's own tolerance. Other-wood "
        "clearance does not depend on L and is reported, not used",
        "isolated_success_by_length": {str(k): v for k, v in iso.items()},
        "curve": curve,
        "chosen_truth_segment_m": chosen,
        "control_at_chosen": summ,
        "isolated_failures_at_chosen": iso_fail,
        "max_cached_graph_vs_tree_model_err_m": wood.max_graph_err_m,
        "provenance": prov,
    }
    path = args.out / "length_calibration.json"
    args.out.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, (json.dumps(out, indent=2) + "\n").encode())
    print(json.dumps({"isolated": out["isolated_success_by_length"], "chosen": chosen}, indent=1))
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg))
    if args.graphs_only:
        return graphs_only(args, cfg)
    I = Isaac(args.isaac_src)
    prov = provenance(args, cfg, I)
    if args.calibrate_length:
        return calibrate(args, I, prov)
    if args.split == "test":
        guard_test(prov, args)
    if TRUTH_SEGMENT_M is None:
        sys.exit(
            "register TRUTH_SEGMENT_M from --calibrate-length before scoring perceived targets"
        )
    cal = args.calibration
    prov["length_calibration"] = {"path": str(cal), "sha256": _sha(cal)}
    if (
        not cal.is_file()
        or json.loads(cal.read_text())["chosen_truth_segment_m"] != TRUTH_SEGMENT_M
    ):
        sys.exit(f"{cal} does not register TRUTH_SEGMENT_M = {TRUTH_SEGMENT_M}")
    li = LENGTH_GRID_M.index(TRUTH_SEGMENT_M)
    frames = EB.frame_list(args)
    cache, wood = GraphCache(args, cfg), Wood(args.cache)
    sens, rows, parity = {}, [], []
    t0 = time.time()
    for n, ref in enumerate(frames):
        fr, par = score_frame(ref, args, I, cache, wood, LENGTH_GRID_M)
        accumulate(sens, fr, len(LENGTH_GRID_M))
        fill_registered(fr, li)
        rows += fr
        parity += par
        print(
            f"[{n + 1}/{len(frames)}] {ref.key} graphs built {cache.built} loaded {cache.loaded} "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    summary = summarize(rows)
    name = "val_dev" if args.split == "val" else "test"
    if args.every != 1 or args.limit:
        name += f"_every{args.every}" + (f"_limit{args.limit}" if args.limit else "")
    result = {
        "what": "offline CPU proxy (Option A): would the Isaac pruner geometry have cut the true "
        "branch at the perceived cut target; a geometry proxy, never an Isaac result",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        **_frames_meta(frames),
        "argv": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "truth_segment_m": TRUTH_SEGMENT_M,
        "summary": summary,
        "length_sensitivity_post_registration": sensitivity(sens, LENGTH_GRID_M),
        "scorer_parity": {
            "frames_x_method_depth": len(parity),
            "all_ok": all(p["ok"] for p in parity),
            "pairs": int(sum(p["pairs"] for p in parity)),
            "scorer_success_jain": int(sum(p["scorer_success_jain"] for p in parity)),
        },
        "max_cached_graph_vs_tree_model_err_m": wood.max_graph_err_m,
        "provenance": prov,
        "pairs": rows_table(rows),
    }
    path = args.out / f"{name}.json"
    _write_atomic(path, (json.dumps(result) + "\n").encode())
    print_table(summary)
    print(f"\nwrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

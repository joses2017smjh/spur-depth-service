"""Build tests/fixtures/k_fix/*.npz: a few hundred GT depth pixels + nearby cylinders.

The fixture lets CI check the intrinsics fix without the NFS dataset:
back-projected GT depth must land on the rendered cylinder surfaces.

python scripts/build_k_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from spur_depth.camera import blender_intrinsics
from spur_depth.data.trunk_stereo_triplet import _euler_xyz_to_T

DATA = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
BARK, TREE, RIG, VIEW = "bark_brown_02", "lpy_envy_00042", "box_cam1", "shot03_l"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "k_fix"


def main() -> int:
    stem = f"{TREE}_{VIEW}"
    ann = json.loads((DATA / "ann" / BARK / TREE / RIG / f"{stem}.json").read_text())
    depth = np.load(DATA / "depth" / BARK / TREE / RIG / f"{stem}.npy")
    mask = np.asarray(Image.open(DATA / "mask" / BARK / TREE / RIG / f"{stem}.png").convert("L"))
    cyl = json.loads((DATA / "cylinders_world" / BARK / f"{TREE}.json").read_text())

    vs, us = np.nonzero(mask > 0)
    pick = np.random.default_rng(0).choice(len(vs), 400, replace=False)
    vs, us = vs[pick], us[pick]
    z = depth[vs, us].astype(np.float64)

    cam = ann["camera"]
    T_wc = _euler_xyz_to_T(cam["location"], cam["rotation_euler"]).astype(np.float64)
    K = blender_intrinsics()
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    R_cw = T_wc[:3, :3].T
    pts_w = (R_cw @ (np.stack([x, y, z], 1) - T_wc[:3, 3]).T).T

    C = np.array([c["centroid"] for c in cyl])
    keep = np.unique(np.concatenate(cKDTree(C).query_ball_point(pts_w, r=0.25)).astype(int))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{TREE}_{RIG}_{VIEW}.npz",
        u=us.astype(np.int16),
        v=vs.astype(np.int16),
        depth=z.astype(np.float32),
        location=np.asarray(cam["location"], dtype=np.float64),
        rotation_euler=np.asarray(cam["rotation_euler"], dtype=np.float64),
        K_annotated=np.asarray(cam["intrinsics"]["K"], dtype=np.float64),
        centroid=C[keep].astype(np.float32),
        orientation=np.array([cyl[i]["orientation"] for i in keep], dtype=np.float32),
        radius=np.array([cyl[i]["radius"] for i in keep], dtype=np.float32),
        length=np.array([cyl[i]["length"] for i in keep], dtype=np.float32),
    )
    print(f"wrote {OUT} with {len(us)} pixels and {len(keep)} cylinders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

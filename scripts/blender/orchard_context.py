"""Orchard context for the Blender tree generator (protocol v4): a full target row plus 3 rows.

The generator renders one L-Py tree on an otherwise empty V-trellis row. Real orchard frames
show neighbouring trees in the same row and further rows behind it, which BranchNet then
calls wood. This module fills the scene around the target tree:

* target row: trees on both faces of the V (the near face leans toward the camera like the
  target, the far face away), one tree per bay along the row;
* ``EXTRA_ROWS`` rows behind it at the measured row spacing, with their posts and wires.

Spacing (metres):
  Envy V-trellis, Prosser WA: row width 3.53 +- 0.03, tree spacing 1.42, 7 wires 46 cm apart,
      canopy angle 75 deg, tree height 3.66 (Davidson et al. 2016, as tabulated in Bhattarai et
      al., arXiv 2304.04919). The L-Py Envy trees span 3.96 m along the row, the template's
      post bay is 3.93 m, so one tree per 3.93 m bay and face.
  UFO sweet cherry, WSU Roza farm, Prosser WA: 3.05 between rows, 1.83 within rows
      (mechanical-harvesting study of 'Selah'/Gisela 6 UFO trees, 2010). The L-Py UFO trees
      span 2.37 m (cordon from the trunk base), so a 2.4 m bay keeps cordons end to end.

Labels: same-row trees get the target's pass index, so they fall inside the rendered tree mask
but on no target cylinder, and the ground truth marks them IGNORE (never trained on, either
way). Trees, posts and wires of the other rows get the background pass index: background,
as in MFO's real labels, where other rows are unlabelled.

Neighbour trees are drawn from the train split of the same kind (never the target), built
once each and instanced (linked mesh data).
"""

from __future__ import annotations

import json
import math
import os
import random

import bpy

ROW_SPACING_M = {"envy": 3.53, "ufo": 3.05}
BAY_M = {"envy": 3.93, "ufo": 2.40}
EXTRA_ROWS = 3
LIBRARY_SIZE = 8
HALF_HFOV_TAN = 960.0 / 1493.3  # half horizontal field of view of the 28 mm render camera
VIEW_MARGIN_M = 3.0  # box_cam poses swing around the target
HELD_OUT = {
    "envy": {f"lpy_envy_{i:05d}" for i in (1, 9, 15, 41, 42, 65)},
    "ufo": {f"lpy_ufo_{i:05d}" for i in range(28, 100)},  # UFO train = 00000-00027
}


def tree_kind(tree_id: str) -> str:
    return "ufo" if "lpy_ufo" in tree_id else "envy"


def reachable_cylinders(meta: dict, cylinders: list) -> list:
    """The cylinders the patched generator renders (hierarchy walk incl. tertiary branches)."""
    hierarchy = meta.get("hierarchy", {})
    keep = set(hierarchy.get("root", [])[:1])
    queue = list(keep)
    while queue:
        parent = queue.pop(0)
        for child in hierarchy.get(parent, []):
            low = child.lower()
            if low.startswith(("branch_", "nontrunk_", "tertiarybranch_", "trunk")):
                keep.add(child)
                queue.append(child)
            elif low.startswith("spur_"):
                keep.add(child)
    return [c for c in cylinders if c.get("part_name") in keep]


def library_mesh(g: dict, tree_id: str):
    path = os.path.join(g["METADATA_DIR"], f"{tree_id}_metadata.json")
    with open(path) as f:
        meta = json.load(f)
    cyl = reachable_cylinders(meta, g["get_cylinder_data_from_metadata"](meta))
    return g["mesh_from_cylinders_local"](cyl, name=f"orchard_lib_{tree_id}")


def bark_material(g: dict, texture_entry: dict):
    diff, normal = g["find_texture_paths"](texture_entry)
    if not diff:
        return None
    return g["make_material_from_textures"](
        diff, normal_path=normal, mat_name=f"Bark_orchard_{texture_entry['name']}"
    )


def _copy(obj, name: str, dy: float, dx: float = 0.0):
    new = obj.copy()  # linked mesh data
    new.name = name
    new.location = (obj.location.x + dx, obj.location.y + dy, obj.location.z)
    bpy.context.collection.objects.link(new)
    return new


def populate(g: dict, tree_obj, tree_id: str, texture_entry: dict) -> dict:
    """Place the orchard around ``tree_obj``; returns the layout record (also for provenance)."""
    kind = tree_kind(tree_id)
    spacing, bay = ROW_SPACING_M[kind], BAY_M[kind]
    rng = random.Random(f"orchard:{tree_id}")
    pool = sorted(
        os.path.basename(p)[: -len("_metadata.json")]
        for p in os.listdir(g["METADATA_DIR"])
        if p.startswith(f"lpy_{kind}_") and p.endswith("_metadata.json")
    )
    pool = [t for t in pool if t != tree_id and t not in HELD_OUT[kind]]
    library_ids = rng.sample(pool, min(LIBRARY_SIZE, len(pool)))
    meshes = [library_mesh(g, t) for t in library_ids]
    mat = bark_material(g, texture_entry)
    for m in meshes:
        tmp = bpy.data.objects.new("orchard_uv_tmp", m)
        bpy.context.collection.objects.link(tmp)
        g["ensure_uv_layer"](tmp)
        bpy.data.objects.remove(tmp, do_unlink=True)
        if mat is not None:
            m.materials.clear()
            m.materials.append(mat)

    base = tree_obj.location.copy()
    tilt_near = g["TREE_TILT_RAD"]
    tilt_far = (-tilt_near[0], tilt_near[1], tilt_near[2])
    cam_dist = abs(g["Y_REF"] - base.y)
    placed = []
    for r in range(EXTRA_ROWS + 1):
        dy = -r * spacing  # rows behind the target, away from the camera
        half_width = (cam_dist + r * spacing) * HALF_HFOV_TAN + VIEW_MARGIN_M
        reach = math.ceil(half_width / bay)
        for face, tilt in (("near", tilt_near), ("far", tilt_far)):
            for j in range(-reach, reach + 1):
                if r == 0 and face == "near" and j == 0:
                    continue  # the target tree itself
                k = rng.randrange(len(meshes))
                jitter = rng.uniform(-0.05, 0.05) * bay if r > 0 else 0.0
                obj = bpy.data.objects.new(f"orchard_r{r}_{face}_{j:+d}", meshes[k])
                bpy.context.collection.objects.link(obj)
                obj.location = (base.x + j * bay + jitter, base.y + dy, base.z)
                obj.rotation_mode = "XYZ"
                obj.rotation_euler = tilt
                obj.pass_index = g["PASS_INDEX_TREE"] if r == 0 else g["PASS_INDEX_BACKGROUND"]
                placed.append(
                    {
                        "tree": library_ids[k],
                        "row": r,
                        "face": face,
                        "bay": j,
                        "location": [round(v, 4) for v in obj.location],
                        "rotation_euler": [round(v, 5) for v in tilt],
                        "pass_index": obj.pass_index,
                    }
                )
    # Trellis furniture: posts every bay along each row, wires copied into the rows behind.
    posts = [o for o in bpy.data.objects if o.name.startswith("post") and o.type == "MESH"]
    wires = [o for o in bpy.data.objects if o.name.startswith("wire") and o.type == "MESH"]
    post_bay = BAY_M["envy"]  # the template's post spacing
    n_furniture = 0
    for r in range(EXTRA_ROWS + 1):
        dy = -r * spacing
        reach = math.ceil(((cam_dist + r * spacing) * HALF_HFOV_TAN + VIEW_MARGIN_M) / post_bay)
        for p in posts:
            for m in range(-reach, reach + 1):
                if r == 0 and m == 0:
                    continue
                c = _copy(p, f"orchard_post_r{r}_{p.name}_{m:+d}", dy, 2 * m * post_bay)
                c.pass_index = g["PASS_INDEX_BACKGROUND"]
                n_furniture += 1
        if r > 0:
            for w in wires:
                c = _copy(w, f"orchard_wire_r{r}_{w.name}", dy)
                c.pass_index = g["PASS_INDEX_BACKGROUND"]
                n_furniture += 1
    bpy.context.view_layer.update()
    return {
        "protocol": "docs/BRANCH_PROTOCOL.md v4",
        "kind": kind,
        "row_spacing_m": spacing,
        "bay_m": bay,
        "extra_rows": EXTRA_ROWS,
        "target": tree_id,
        "target_location": [round(v, 4) for v in base],
        "library": library_ids,
        "trees": placed,
        "furniture_copies": n_furniture,
    }

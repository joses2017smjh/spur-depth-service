"""CPU preview of the orchard context (protocol v4): two low-resolution views, no dataset output.

blender -b orchard_template.blend -P preview_orchard.py   (env: GEN, TREE, OUT, SAMPLES)
GEN is the patched generator copy (orchard_context.py next to it); OUT a directory.
"""

import json
import math
import os
import runpy
import sys

import bpy

gen = os.environ["GEN"]
sys.path.insert(0, os.path.dirname(gen))
import orchard_context  # noqa: E402

g = runpy.run_path(gen, run_name="orchard_preview")  # module globals, main() not run
tree_id, out = os.environ["TREE"], os.environ["OUT"]
os.makedirs(out, exist_ok=True)
scene = bpy.context.scene
g["remove_placeholder_objects"]()
g["fix_world_background"]()
g["fix_ground_material"]()
cam = bpy.data.objects[g["CAM"]]
cam.data.lens = 28.0
scene.camera = cam
g["render_tree"](scene)
tex = next(t for t in g["BARK_TEXTURES"] if t["name"] == "bark_brown_02")
template = bpy.data.objects.get("tree0_TRUNK")
saved = {0: (template.matrix_world.to_translation().copy(), None)} if template else {}
g["remove_all_tree_objects"]()
meta = os.path.join(g["METADATA_DIR"], f"{tree_id}_metadata.json")
tree_obj = g["get_tree_object_for_metadata"](tree_id, meta, 0, saved, tex)
layout = orchard_context.populate(g, tree_obj, tree_id, tex)
with open(os.path.join(out, f"orchard_{tree_id}.json"), "w") as f:
    json.dump(layout, f, indent=1)
scene.render.resolution_x, scene.render.resolution_y = 960, 540
scene.cycles.samples = int(os.environ.get("SAMPLES", "24"))
scene.cycles.use_denoising = True
scene.cycles.device = "CPU"
z2 = g["get_z_levels"]()[1]
views = {
    "camera": ((g["X_REF"], g["Y_REF"], z2), g["ROTATION_REF"]),
    # High and behind the camera, looking over the target row toward the rows behind it.
    "overview": (
        (g["X_REF"], g["Y_REF"] + 6.0, 9.0),
        (math.radians(58), 0.0, g["ROTATION_REF"][2]),
    ),
}
for name, (loc, rot) in views.items():
    cam.location = loc
    cam.rotation_euler = rot
    scene.render.filepath = os.path.join(out, f"{tree_id}_{name}.png")
    bpy.ops.render.render(write_still=True)
print("preview done", tree_id, len(layout["trees"]), "trees")

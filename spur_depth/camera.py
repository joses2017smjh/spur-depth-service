"""Pinhole intrinsics for the Blender orchard renders.

Every ``ann/*.json`` stores ``camera.intrinsics.K`` as the hard-coded
``K_REF`` from ``Dataloader/generate_tree2.py`` (``[[2666.67, 0, 960],
[0, 1500, 540]]``). The render camera is a 28 mm lens on Blender's default
36 mm sensor (``cam_obj.data.lens = 28.0``), so the true focal length is
``28 / 36 * 1920 = 1493.33`` px on both axes.

Measured, not assumed: back-projecting GT depth with ``f = 1493.33`` and the
principal point at ``(959.5, 539.5)`` puts mask pixels on the rendered
cylinder surfaces with a median error of about 0.09 mm (7 frames, 5 trees,
all rigs, both stereo sides; a free fit gives f = 1493.45 +- 0.10). The
annotated ``K`` misses the same surfaces by a median of 13-17 cm.

Depth ``.npy`` files are planar z-depth, not ray length.
"""

from __future__ import annotations

import warnings

import numpy as np

RENDER_LENS_MM = 28.0
SENSOR_WIDTH_MM = 36.0

# The value written into every annotation. Not the render camera.
K_REF_ANNOTATED = np.array(
    [[2666.666666666667, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def blender_intrinsics(
    width: int = 1920,
    height: int = 1080,
    lens_mm: float = RENDER_LENS_MM,
    sensor_width_mm: float = SENSOR_WIDTH_MM,
) -> np.ndarray:
    """3x3 K for a Blender camera with ``sensor_fit='AUTO'`` and square pixels.

    The principal point is in pixel-index coordinates (pixel centres at
    integers), so it sits at ``(W - 1) / 2, (H - 1) / 2``.
    """
    f = float(lens_mm) / float(sensor_width_mm) * float(max(width, height))
    return np.array(
        [[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def is_annotated_k_ref(K) -> bool:
    """True when ``K`` is the known-wrong annotation matrix."""
    K = np.asarray(K, dtype=np.float64)
    return K.shape == (3, 3) and bool(np.allclose(K, K_REF_ANNOTATED, atol=1e-3))


def intrinsics_from_ann(ann: dict, *, trust_annotation: bool = False) -> np.ndarray:
    """Return the render camera's K for one annotation.

    ``trust_annotation=True`` returns the stored matrix unchanged, which only
    reproduces numbers computed before this fix.
    """
    intr = ann["camera"]["intrinsics"]
    K = np.asarray(intr["K"], dtype=np.float64)
    if trust_annotation:
        return K
    if is_annotated_k_ref(K):
        width = int(intr.get("width", 1920))
        height = int(intr.get("height", 1080))
        return blender_intrinsics(width, height)
    warnings.warn(
        "annotation K is not the known K_REF; using it as stored", RuntimeWarning, stacklevel=2
    )
    return K


def scale_intrinsics(K, sx: float, sy: float | None = None) -> np.ndarray:
    """K for an image resized by ``sx`` (and ``sy``) under the pixel-centre convention."""
    sy = sx if sy is None else sy
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K

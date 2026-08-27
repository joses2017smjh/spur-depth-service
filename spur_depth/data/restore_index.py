"""Index the 2026-08-20 val-only restore of ``Data/full_spur``.

Nine trees, 60 DA2-ft frames each. The 24k-frame train set is still missing.
Paper val trees are ``lpy_envy_00042`` and ``lpy_envy_00065``; everything
else is fair game to fit on.
"""

from __future__ import annotations

from pathlib import Path

PAPER_VAL = ("lpy_envy_00042", "lpy_envy_00065")
DEFAULT_ROOT = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
BARK = "bark_brown_02"


def discover_da2_triples(root: Path | None = None) -> list[dict]:
    """Return rows with pred/gt/mask/rgb/ann for every restored DA2-ft frame."""
    root = Path(root or DEFAULT_ROOT)
    bark = root / "Da2Finetune" / BARK
    rows: list[dict] = []
    if not bark.is_dir():
        return rows
    for tree_dir in sorted(p for p in bark.iterdir() if p.is_dir()):
        tree = tree_dir.name
        for npy in sorted(tree_dir.rglob("*.npy")):
            rel = npy.relative_to(tree_dir)
            gt = root / "depth" / BARK / tree / rel
            mask = root / "mask" / BARK / tree / rel.with_suffix(".png")
            rgb = root / "Optical_flow" / BARK / tree / rel.with_suffix(".png")
            if not (gt.is_file() and mask.is_file()):
                continue
            parts = npy.stem.split("_")
            side = parts[-1]
            shot = parts[-2]
            box_dir = root / "box_mask" / BARK / tree / rel.parts[0]
            box_mask = None
            if box_dir.is_dir():
                hits = sorted(box_dir.glob(f"{tree}_{shot}_{side}_box_*.png"))
                box_mask = hits[0] if hits else None
            rows.append(
                {
                    "tree": tree,
                    "set_id": rel.parts[0],
                    "shot": shot,
                    "side": side,
                    "pred": npy,
                    "gt": gt,
                    "mask": mask,
                    "rgb": rgb if rgb.is_file() else None,
                    "ann": root / "ann" / BARK / tree / rel.with_suffix(".json"),
                    "box_mask": box_mask,
                }
            )
    return rows


def discover_train_or_restore(root: Path | None = None) -> list[dict]:
    """Same indexer. 24k train appears as more trees under Da2Finetune/.

    Extra roots: ``SPUR_TRAIN_ROOT`` (Depot, when allocated). Refuses to
    invent frames. Callers that need orchard-scale must check ``len``.
    """
    import os

    rows = discover_da2_triples(root)
    extra = os.environ.get("SPUR_TRAIN_ROOT")
    if extra:
        more = discover_da2_triples(Path(extra))
        seen = {str(r["pred"]) for r in rows}
        rows.extend(r for r in more if str(r["pred"]) not in seen)
    return rows


def split_holdout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fit on the seven non-paper trees; report on ``PAPER_VAL``."""
    hold = [r for r in rows if r["tree"] in PAPER_VAL]
    fit = [r for r in rows if r["tree"] not in PAPER_VAL]
    return fit, hold

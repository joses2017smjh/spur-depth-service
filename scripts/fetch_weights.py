#!/usr/bin/env python3
"""Fetch the refiner checkpoint into /weights and verify SHA256.

Weights are not baked into the image. Mount a volume, set SPUR_CKPT, or
pass --release-url once a GitHub Release exists. Until then this script
copies from a local path and checks the digest in weights.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LOCK = _REPO / "weights.lock"


def _expected() -> tuple[str, str, int]:
    for line in _LOCK.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sha, size, *rest = line.split()
        return name, sha, int(size)
    raise SystemExit("weights.lock has no entries")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="local checkpoint path")
    p.add_argument("--dst", type=str, default="/weights/refiner.pt")
    args = p.parse_args(argv)

    name, expect, size = _expected()
    src = Path(args.src)
    if not src.is_file():
        print(f"missing source {src}", file=sys.stderr)
        return 2
    digest = _sha256(src)
    if digest != expect:
        print(
            f"SHA256 mismatch for {name}: got {digest}, expected {expect}",
            file=sys.stderr,
        )
        return 1
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"ok {name} {digest} -> {dst} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

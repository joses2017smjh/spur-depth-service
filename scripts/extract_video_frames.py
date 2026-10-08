"""Unlabelled real frames for adaptation (protocol v5): every Nth frame of MFO source videos.

The UFO cherry source videos (MFO "Labelled Data/UFORozaVideos") hold RGB and colour-mapped
depth side by side (1280x480); the left half is kept. Only videos of MFO's train split
(89-163) are used, so MFO val and test videos stay unseen.

python scripts/extract_video_frames.py VIDEO_DIR OUT_DIR --videos 89-163 --stride 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--videos", dest="videos_range", default="89-163", help="inclusive id range")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args(argv)
    lo, hi = (int(v) for v in args.videos_range.split("-"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, paths = {"stride": args.stride, "range": [lo, hi], "videos": {}}, []
    for f in sorted(args.videos.glob("video_*.avi"), key=lambda p: int(p.stem.split("_")[1])):
        vid = int(f.stem.split("_")[1])
        if not lo <= vid <= hi:
            continue
        cap = cv2.VideoCapture(str(f))
        n, kept = 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % args.stride == 0:
                left = frame[:, : frame.shape[1] // 2]
                p = args.out / f"video_{vid}" / f"{n:05d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), left)
                kept.append(n)
                paths.append(str(p))
            n += 1
        cap.release()
        manifest["videos"][f.name] = {
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "frames": n,
            "kept": kept,
        }
        print(f.name, n, "frames,", len(kept), "kept", flush=True)
    (args.out / "frames.txt").write_text("\n".join(paths) + "\n")
    manifest["n_frames"] = len(paths)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("total", len(paths), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

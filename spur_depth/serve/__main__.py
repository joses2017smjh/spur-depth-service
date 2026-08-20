"""``python -m spur_depth.serve`` — one worker, on purpose.

Four processes each build a ViT-L context and each hold a copy of the
weights; the GPU serialises them anyway. One worker + the asyncio lock in
``app.py`` is the concurrency model.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("SPUR_HOST", "0.0.0.0")
    port = int(os.environ.get("SPUR_PORT", "8000"))
    uvicorn.run(
        "spur_depth.serve.app:app",
        host=host,
        port=port,
        workers=1,
        log_level=os.environ.get("SPUR_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()

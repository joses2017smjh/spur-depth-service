"""FastAPI service: single-view DA2-ft path + 6-view DINO refinement path.

Concurrency: one worker, one GPU lock, bounded waiters, 503 when full.
Do not run ``uvicorn --workers 4`` — four processes each hold a copy of
ViT-L and the GPU serialises them anyway.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from spur_depth.serve import metrics
from spur_depth.serve.drift import DriftMonitor
from spur_depth.serve.input_stats import request_stats
from spur_depth.serve.runner import N_VIEWS, build_runner
from spur_depth.serve.schemas import LatencyMs, ModelRef, PredictResponse

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_REPO = Path(__file__).resolve().parents[2]
_MODEL_CARD = _REPO / "model_card.json"


class InferenceGate:
    """Admit at most ``max_admitted`` requests (in-flight + waiting).

    One GPU, one in-flight inference. Extra waiters sit on ``_lock``;
    beyond ``max_admitted`` we fail 503 rather than unbounded-queueing a
    489 ms model.
    """

    def __init__(self, max_admitted: int = 4) -> None:
        self._lock = asyncio.Lock()
        self._state = asyncio.Lock()
        self._admitted = 0
        self._max = max_admitted

    async def __aenter__(self) -> "InferenceGate":
        async with self._state:
            if self._admitted >= self._max:
                raise HTTPException(status_code=503, detail="inference queue full")
            self._admitted += 1
        await self._lock.acquire()
        return self

    async def __aexit__(self, *exc) -> None:
        self._lock.release()
        async with self._state:
            self._admitted -= 1


def _encode_npy(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr.astype(np.float32)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _read_image(raw: bytes, filename: str) -> Image.Image:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    if not raw:
        raise HTTPException(status_code=422, detail=f"{filename}: empty upload")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{filename}: not a readable image ({exc})"
        ) from exc
    return img.convert("RGB")


def _read_depth(raw: bytes, filename: str) -> np.ndarray:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        arr = np.load(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"{filename}: not a readable .npy ({exc})"
        ) from exc
    return np.asarray(arr, dtype=np.float32)


def _latency(pre: float, infer: float, post: float) -> LatencyMs:
    return LatencyMs(pre=pre, infer=infer, post=post, total=pre + infer + post)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runner = build_runner()
        app.state.gate = InferenceGate(max_admitted=int(os.environ.get("SPUR_MAX_QUEUE", "4")))
        app.state.drift = DriftMonitor()
        app.state.runner.warmup()
        app.state.ready = True
        yield

    app = FastAPI(
        title="SPUR metric depth",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ready", "backend": app.state.runner.info.backend}

    @app.get("/model")
    async def model_card():
        if _MODEL_CARD.is_file():
            return json.loads(_MODEL_CARD.read_text())
        info = app.state.runner.info
        return {
            "model": "MVStereoDINOUNet",
            "checkpoint_sha256_prefix": info.sha,
            "precision": info.precision,
            "backend": info.backend,
            "train_domain": (
                "synthetic Blender, bark_brown_02, dormant Envy/UFO — "
                "this model has never seen a real orchard"
            ),
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        payload, content_type = metrics.render()
        return Response(content=payload, media_type=content_type)

    @app.get("/drift")
    async def drift_snapshot():
        return app.state.drift.snapshot()

    def _model_ref() -> ModelRef:
        info = app.state.runner.info
        return ModelRef(sha=info.sha, precision=info.precision, backend=info.backend)  # type: ignore[arg-type]

    @app.post("/predict", response_model=PredictResponse)
    async def predict(image: UploadFile = File(...)):
        raw = await image.read()
        t0 = time.perf_counter()
        img = _read_image(raw, image.filename or "image")
        pre = (time.perf_counter() - t0) * 1e3
        async with app.state.gate:
            t1 = time.perf_counter()
            try:
                depth = app.state.runner.predict_one(img)
            except NotImplementedError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            infer = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        payload = _encode_npy(depth)
        post = (time.perf_counter() - t2) * 1e3
        lat = _latency(pre, infer, post)
        metrics.observe("/predict", lat.model_dump())
        snap = app.state.drift.observe(request_stats(images=img, depths=depth))
        metrics.observe_drift(snap)
        return PredictResponse(
            depth_npy_b64=payload,
            shape=list(depth.shape),
            latency_ms=lat,
            model=_model_ref(),
        )

    @app.post("/predict/group", response_model=PredictResponse)
    async def predict_group(
        images: list[UploadFile] = File(...),
        group_id: str = Form("ungrouped"),
        depths: list[UploadFile] | None = File(None),
    ):
        n = len(images)
        if n != N_VIEWS:
            raise HTTPException(
                status_code=422,
                detail=f"expected {N_VIEWS} views, got {n}",
            )
        t0 = time.perf_counter()
        pil = [
            _read_image(await im.read(), im.filename or f"image{i}") for i, im in enumerate(images)
        ]
        depth_arrs = None
        if depths:
            if len(depths) != N_VIEWS:
                raise HTTPException(
                    status_code=422,
                    detail=f"expected {N_VIEWS} depth maps, got {len(depths)}",
                )
            depth_arrs = [
                _read_depth(await d.read(), d.filename or f"depth{i}") for i, d in enumerate(depths)
            ]
        pre = (time.perf_counter() - t0) * 1e3
        async with app.state.gate:
            t1 = time.perf_counter()
            try:
                stacked = app.state.runner.predict_group(pil, depth_arrs)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            infer = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        payload = _encode_npy(stacked)
        post = (time.perf_counter() - t2) * 1e3
        lat = _latency(pre, infer, post)
        metrics.observe("/predict/group", lat.model_dump())
        snap = app.state.drift.observe(request_stats(images=pil, depths=stacked, group_id=group_id))
        metrics.observe_drift(snap)
        return PredictResponse(
            depth_npy_b64=payload,
            shape=list(stacked.shape),
            latency_ms=lat,
            model=_model_ref(),
            group_id=group_id,
        )

    @app.post("/branches")
    async def branches(
        image: UploadFile = File(...),
        depth: UploadFile = File(...),
        fx: float = Form(...),
        fy: float = Form(...),
        cx: float = Form(...),
        cy: float = Form(...),
        mono_depth: UploadFile | None = File(None),
    ):
        """One RGB-D frame -> tree graph (parts, parents, junctions, cut points), metres."""
        from spur_depth.branches.service import BranchService

        if not hasattr(app.state, "branches"):
            app.state.branches = BranchService.from_env()
        rgb = np.asarray(_read_image(await image.read(), image.filename or "image"))
        z = _read_depth(await depth.read(), depth.filename or "depth")
        mono = None
        if mono_depth is not None:
            mono = _read_depth(await mono_depth.read(), mono_depth.filename or "mono_depth")
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        async with app.state.gate:
            try:
                return app.state.branches.run(rgb, z, K, mono)
            except NotImplementedError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()

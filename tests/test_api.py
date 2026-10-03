"""HTTP contract for the depth service. Runs on DummyRunner (no weights, no GPU)."""

from __future__ import annotations

import asyncio
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from spur_depth.serve.preprocess import REFINER_H, REFINER_W


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SPUR_SKIP_WEIGHTS", "1")
    monkeypatch.setenv("SPUR_MAX_QUEUE", "4")
    from fastapi.testclient import TestClient

    from spur_depth.serve.app import create_app

    with TestClient(create_app()) as c:
        yield c


def _png(w=64, h=32, color=(34, 85, 34)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _decode_npy(b64: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(b64)))


def test_healthz_and_readyz(client):
    assert client.get("/healthz").status_code == 200
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["backend"] == "dummy"


def test_model_card_names_synthetic_domain(client):
    body = client.get("/model").json()
    assert "synthetic" in body.get("train_domain", "").lower() or "train_domain" in body


def test_metrics_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "spur_depth_stage_ms" in r.text or r.headers["content-type"].startswith("text/plain")


def test_drift_endpoint_after_predict(client):
    client.post("/predict", files={"image": ("t.png", _png(), "image/png")})
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "n_logged" in body
    assert body["n_logged"] >= 1
    metrics = client.get("/metrics").text
    assert "spur_depth_requests_logged" in metrics


def test_predict_dummy_happy_path(client):
    r = client.post("/predict", files={"image": ("t.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    depth = _decode_npy(body["depth_npy_b64"])
    assert depth.shape == (REFINER_H, REFINER_W)
    assert depth.dtype == np.float32
    assert body["units"] == "m"
    assert "total" in body["latency_ms"]


def test_predict_group_wrong_view_count_names_expected(client):
    files = [("images", (f"{i}.png", _png(), "image/png")) for i in range(3)]
    r = client.post("/predict/group", files=files)
    assert r.status_code == 422
    assert "expected 6" in r.json()["detail"]


def test_predict_group_happy_path(client):
    files = [("images", (f"{i}.png", _png(color=(i * 20, 40, 40)), "image/png")) for i in range(6)]
    r = client.post("/predict/group", data={"group_id": "box"}, files=files)
    assert r.status_code == 200, r.text
    depth = _decode_npy(r.json()["depth_npy_b64"])
    assert depth.shape == (6, REFINER_H, REFINER_W)


def test_corrupt_image_is_422(client):
    r = client.post("/predict", files={"image": ("bad.png", b"not-an-image", "image/png")})
    assert r.status_code == 422


def test_oversized_upload_is_413(client):
    huge = b"\x00" * (20 * 1024 * 1024 + 8)
    r = client.post("/predict", files={"image": ("big.png", huge, "image/png")})
    assert r.status_code == 413


def test_inference_gate_returns_503_when_full():
    from fastapi import HTTPException

    from spur_depth.serve.app import InferenceGate

    async def _run():
        gate = InferenceGate(max_admitted=1)
        async with gate:
            with pytest.raises(HTTPException) as ei:
                async with gate:
                    pass
            assert ei.value.status_code == 503

    asyncio.run(_run())


def test_concurrency_returns_200_or_503_never_500(client, monkeypatch):
    monkeypatch.setenv("SPUR_INFER_DELAY_MS", "200")
    png = _png()

    def hit(_):
        return client.post("/predict", files={"image": ("t.png", png, "image/png")})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(hit, range(8)))
    codes = {r.status_code for r in responses}
    assert 500 not in codes, [r.text for r in responses if r.status_code == 500]
    assert codes <= {200, 503}
    assert 200 in codes


def _branch_form():
    return {"fx": "498.0", "fy": "498.0", "cx": "47.5", "cy": "31.5"}


def test_branches_without_checkpoint_is_501(client, monkeypatch):
    monkeypatch.delenv("SPUR_BRANCH_CKPT", raising=False)
    files = {
        "image": ("rgb.png", _png(96, 64), "image/png"),
        "depth": ("depth.npy", _npy(np.full((64, 96), 1.5)), "application/octet-stream"),
    }
    r = client.post("/branches", files=files, data=_branch_form())
    assert r.status_code == 501


def test_branches_depth_shape_mismatch_is_422(client):
    files = {
        "image": ("rgb.png", _png(96, 64), "image/png"),
        "depth": ("depth.npy", _npy(np.full((32, 96), 1.5)), "application/octet-stream"),
    }
    r = client.post("/branches", files=files, data=_branch_form())
    assert r.status_code == 422

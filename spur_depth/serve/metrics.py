"""Prometheus histograms for the two serving endpoints.

Buckets straddle the measured PyTorch numbers on the Quadro RTX 8000
(refiner fp32 p50 ≈ 489 ms, fp16 p50 ≈ 185 ms) and the paper's ~80 ms/view
DA2-ft path, so p50/p95/p99 are actually on the histogram rather than
saturating a generic 1 s bucket.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest

_BUCKETS = (5, 10, 20, 40, 80, 160, 320, 640, 1280)

INFER_MS = Histogram(
    "spur_depth_stage_ms",
    "Per-request stage latency in milliseconds",
    labelnames=("endpoint", "stage"),
    buckets=_BUCKETS,
)

DRIFT_PSI = Gauge(
    "spur_depth_drift_psi",
    "PSI of the rolling request window vs the committed baseline",
    labelnames=("feature",),
)

DRIFT_KS = Gauge(
    "spur_depth_drift_ks_d",
    "Two-sample KS D of the rolling window vs the committed baseline",
    labelnames=("feature",),
)

DRIFT_ALERT = Gauge(
    "spur_depth_drift_alert",
    "1 if any feature's PSI exceeds 0.25",
)

REQUESTS_LOGGED = Gauge("spur_depth_requests_logged", "Request-stat rows appended")


def observe(endpoint: str, latency_ms: dict) -> None:
    for stage in ("pre", "infer", "post", "total"):
        if stage in latency_ms:
            INFER_MS.labels(endpoint=endpoint, stage=stage).observe(latency_ms[stage])


def observe_drift(snapshot: dict) -> None:
    REQUESTS_LOGGED.set(snapshot.get("n_logged", 0))
    DRIFT_ALERT.set(1.0 if snapshot.get("alert") else 0.0)
    for feat, row in (snapshot.get("features") or {}).items():
        if "psi" in row:
            DRIFT_PSI.labels(feature=feat).set(row["psi"])
        if "ks_d" in row:
            DRIFT_KS.labels(feature=feat).set(row["ks_d"])


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

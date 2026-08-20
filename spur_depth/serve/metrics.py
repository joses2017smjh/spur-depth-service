"""Prometheus histograms for the two serving endpoints.

Buckets straddle the measured PyTorch numbers on the Quadro RTX 8000
(refiner fp32 p50 ≈ 489 ms, fp16 p50 ≈ 185 ms) and the paper's ~80 ms/view
DA2-ft path, so p50/p95/p99 are actually on the histogram rather than
saturating a generic 1 s bucket.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest

_BUCKETS = (5, 10, 20, 40, 80, 160, 320, 640, 1280)

INFER_MS = Histogram(
    "spur_depth_stage_ms",
    "Per-request stage latency in milliseconds",
    labelnames=("endpoint", "stage"),
    buckets=_BUCKETS,
)


def observe(endpoint: str, latency_ms: dict) -> None:
    for stage in ("pre", "infer", "post", "total"):
        if stage in latency_ms:
            INFER_MS.labels(endpoint=endpoint, stage=stage).observe(latency_ms[stage])


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

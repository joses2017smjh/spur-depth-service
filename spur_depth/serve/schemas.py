"""Pydantic request/response models. ``/docs`` is the API reference."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LatencyMs(BaseModel):
    pre: float
    infer: float
    post: float
    total: float


class ModelRef(BaseModel):
    sha: str
    precision: Literal["fp32", "fp16", "dummy"]
    backend: str


class PredictResponse(BaseModel):
    depth_npy_b64: str = Field(
        description="Raw float32 .npy bytes, base64. Metres. Not a 16-bit PNG."
    )
    shape: List[int]
    dtype: Literal["float32"] = "float32"
    units: Literal["m"] = "m"
    latency_ms: LatencyMs
    model: ModelRef
    group_id: Optional[str] = None

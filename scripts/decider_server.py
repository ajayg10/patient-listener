"""Minimal FastAPI server exposing Strands Decider over HTTP.

Designed for execution on local machines or remote GPU notebooks (Google Colab / Kaggle).
Endpoints:
- POST /decide: Evaluates a state against a structured ChoiceQuestion.
- GET /health: Health probe reporting active hardware device and model ID.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from strands_decider.schema import ChoiceQuestion

from backend.app.config import get_config
from backend.app.decider_service import LocalDecider

# Auto-select CUDA if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

decider: LocalDecider | None = None


class DecideRequest(BaseModel):
    state: str = Field(..., description="Conversational state context text")
    question: dict[str, Any] = Field(..., description="ChoiceQuestion specification (instructions, criteria)")


class DecideResponse(BaseModel):
    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    backend: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global decider
    cfg = get_config()
    # Override device with auto-detected hardware accelerator
    cfg.decider_device = DEVICE
    # If HF_HOME is set in env, preserve it
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = cfg.hf_home

    print(f"[decider_server] Initializing LocalDecider on device={DEVICE}...")
    decider = LocalDecider(cfg=cfg)
    print(f"[decider_server] Ready to serve requests on device={DEVICE}.")
    yield
    print("[decider_server] Shutting down.")


app = FastAPI(
    title="Patient Listener Decider Server",
    description="Minimal inference server wrapping Strands Decider 2B",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    cfg = get_config()
    return {
        "status": "ok",
        "device": DEVICE,
        "model": cfg.decider_model_id,
        "backend": "local",
    }


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    global decider
    if decider is None:
        raise HTTPException(status_code=503, detail="Decider engine is not initialized.")

    try:
        q = ChoiceQuestion(
            instructions=req.question.get("instructions", ""),
            criteria=req.question.get("criteria", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid question schema: {e}") from e

    decision = decider.decide(state_text=req.state, question=q)

    return DecideResponse(
        choice=decision.choice,
        probabilities=decision.probabilities,
        confidence=decision.confidence,
        latency_ms=decision.latency_ms,
        backend=decision.backend,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

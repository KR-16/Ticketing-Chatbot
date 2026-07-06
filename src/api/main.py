"""FastAPI inference service.

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

The model bundle directory defaults to models/bundle and can be overridden
with the BUNDLE_DIR environment variable. The bundle is loaded once at
startup; if loading fails the API stays up and reports the problem via
/health (503) instead of crashing, so orchestrators can see *why*.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, PredictionResponse, TicketRequest
from src.inference.predictor import TicketPredictor

logger = logging.getLogger(__name__)


def _load_predictor() -> TicketPredictor:
    bundle_dir = os.getenv("BUNDLE_DIR", "models/bundle")
    return TicketPredictor(bundle_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = _load_predictor()
        app.state.load_error = None
    except Exception as error:
        logger.error(
            f"Model bundle failed to load: {error}. "
            "Train one (python main.py or the Colab notebook) or set BUNDLE_DIR."
        )
        app.state.predictor = None
        app.state.load_error = str(error)
    yield


app = FastAPI(
    title="Ticketing-Chatbot API",
    description="Classifies support tickets into Change / Incident / "
                "Problem / Request.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a traceback to callers; log it, return clean JSON."""
    logger.exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"}
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded: {request.app.state.load_error}. "
                "Train a bundle (python main.py or notebooks/train_colab.ipynb) "
                "or point BUNDLE_DIR at one."
            ),
        )
    return HealthResponse(status="ok", classes=predictor.classes)


@app.post("/predict", response_model=PredictionResponse)
def predict(ticket: TicketRequest, request: Request) -> PredictionResponse:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {request.app.state.load_error}",
        )
    try:
        result = predictor.predict(subject=ticket.subject, body=ticket.body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return PredictionResponse(**result)

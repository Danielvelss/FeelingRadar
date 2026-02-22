# -*- coding: utf-8 -*-
"""
RoBERTuito Sentiment API — FastAPI service for EC2.

Run on EC2:
    MODEL_PATH=/home/ubuntu/robertuito-model uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_PATH  — path to the model directory (default: ./model)
    DEVICE      — -1 for CPU, 0 for GPU (default: -1)
"""

from contextlib import asynccontextmanager
from typing import List
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline
from pysentimiento.preprocessing import preprocess_tweet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LABEL_MAPPING = {
    "positivo": "positive",
    "negativo": "negative",
    "neutro": "neutral",
}

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    model_path = os.getenv("MODEL_PATH", "./model")
    device = int(os.getenv("DEVICE", "-1"))
    logger.info(f"Loading model from {model_path} on device {device}...")
    classifier = hf_pipeline(
        "text-classification",
        model=model_path,
        device=device,
    )
    logger.info("Model loaded successfully")
    yield
    classifier = None


app = FastAPI(title="RoBERTuito Sentiment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Schemas ----

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    score: float

class BatchRequest(BaseModel):
    texts: List[str]

class BatchResponse(BaseModel):
    results: List[PredictResponse]


# ---- Endpoints ----

@app.get("/health")
def health():
    return {
        "status": "ok" if classifier else "loading",
        "model": "robertuito-guatemala-v2.0",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not classifier:
        raise HTTPException(503, "Model not loaded")
    try:
        text_prep = preprocess_tweet(req.text)
        result = classifier(text_prep, truncation=True, max_length=128)
        sentiment = LABEL_MAPPING.get(result[0]["label"], "neutral")
        score = round(result[0]["score"], 4)
        return PredictResponse(sentiment=sentiment, score=score)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not classifier:
        raise HTTPException(503, "Model not loaded")
    results = []
    for text in req.texts:
        try:
            text_prep = preprocess_tweet(text)
            result = classifier(text_prep, truncation=True, max_length=128)
            sentiment = LABEL_MAPPING.get(result[0]["label"], "neutral")
            score = round(result[0]["score"], 4)
            results.append(PredictResponse(sentiment=sentiment, score=score))
        except Exception as e:
            logger.warning(f"Error processing text: {e}")
            results.append(PredictResponse(sentiment="neutral", score=0.0))
    return BatchResponse(results=results)

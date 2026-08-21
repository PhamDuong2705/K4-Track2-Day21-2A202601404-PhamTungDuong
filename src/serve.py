"""FastAPI inference service for the Adult Income model stored on Amazon S3."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

import boto3
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field


MODEL_KEY = os.getenv("MODEL_KEY", "artifacts/current/model.joblib")
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(Path.home() / "models" / "model.joblib"))
).expanduser()
FEATURE_COUNT = 10

model: Any | None = None


def download_model() -> Path:
    """Download the current model from S3 using the EC2 IAM role or AWS env vars."""
    bucket = os.getenv("ARTIFACT_BUCKET")
    if not bucket:
        raise RuntimeError("ARTIFACT_BUCKET environment variable is required")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(f"{MODEL_PATH.suffix}.download")
    boto3.client("s3").download_file(bucket, MODEL_KEY, str(temporary_path))
    temporary_path.replace(MODEL_PATH)
    print(f"Downloaded s3://{bucket}/{MODEL_KEY} to {MODEL_PATH}")
    return MODEL_PATH


def load_model() -> Any:
    """Download and load the production model."""
    global model
    model = joblib.load(download_model())
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Income Model API", version="1.0.0", lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description=(
            "age, workclass, education_num, marital_status, occupation, "
            "relationship, sex, capital_gain, capital_loss, hours_per_week"
        ),
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Report readiness only after the model has been loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest) -> dict[str, int | str]:
    """Predict whether annual income is above USD 50K."""
    if len(req.features) != FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {FEATURE_COUNT} features, got {len(req.features)}",
        )
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}
    if prediction not in labels:
        raise HTTPException(status_code=500, detail="Model returned an invalid class")
    return {"prediction": prediction, "label": labels[prediction]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

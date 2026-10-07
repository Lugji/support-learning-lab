from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Support Learning Lab")

model_path = Path(__file__).parent / "artifacts" / "baseline_model.joblib"
model = joblib.load(model_path)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="Text darf nicht leer sein.")

    probabilities = model.predict_proba([text])[0]
    top_indices = probabilities.argsort()[-3:][::-1]

    return {
        "text": text,
        "predictions": [
            {
                "category": str(model.classes_[index]),
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ],
    }


import csv
from typing import Literal


@app.get("/experiments")
def get_experiments(weighting: Literal["none", "balanced"] = "balanced"):
    reports_dir = Path(__file__).parent / "reports"
    if weighting == "balanced":
        reports_dir = reports_dir / "balanced"

    report_path = reports_dir / "active_learning_summary.csv"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Die Ergebnisse für diese Gewichtung fehlen.",
        )

    with report_path.open(newline="", encoding="utf-8") as file:
        results = [
            {
                "strategy": row["strategy"],
                "labeled_examples": int(row["labeled_examples"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
            for row in csv.DictReader(file)
        ]

    return {
        "dataset": "BANKING77",
        "metric": "macro_f1",
        "evaluation": "validation",
        "weighting": weighting,
        "seeds": [7, 21, 42],
        "results": results,
    }


from annotation_api import router as annotation_router

app.include_router(annotation_router)


from annotation_status import router as status_router

app.include_router(status_router)

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

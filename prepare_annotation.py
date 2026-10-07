from contextlib import closing
from pathlib import Path
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "annotations.sqlite3"

if DB_PATH.exists():
    raise SystemExit(
        "Die Datenbank existiert bereits. Vorhandene Annotationen bleiben erhalten."
    )

df = pd.read_csv(ROOT / "data" / "train.csv")
train, validation = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["category"]
)
assert set(train.index).isdisjoint(validation.index)

rng = np.random.default_rng(7)
initial_ids = []

for category in sorted(train["category"].unique()):
    ids = train.index[train["category"] == category].to_numpy()
    initial_ids.extend(rng.choice(ids, size=5, replace=False))

pool_ids = np.setdiff1d(train.index.to_numpy(), initial_ids)
assert set(initial_ids).isdisjoint(pool_ids)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    )),
])

print("Trainiere das Annotation-Modell mit 385 Beispielen …", flush=True)
model.fit(
    train.loc[initial_ids, "text"],
    train.loc[initial_ids, "category"],
)

print("Berechne die Unsicherheit der übrigen Texte …", flush=True)
probabilities = model.predict_proba(train.loc[pool_ids, "text"])
records = []

for sample_id, scores in zip(pool_ids, probabilities):
    top = scores.argsort()[-3:][::-1]
    predictions = [
        {
            "category": str(model.classes_[index]),
            "probability": float(scores[index]),
        }
        for index in top
    ]
    margin = float(scores[top[0]] - scores[top[1]])
    records.append((
        int(sample_id),
        str(train.loc[sample_id, "text"]),
        margin,
        json.dumps(predictions),
    ))

with closing(sqlite3.connect(DB_PATH)) as connection:
    connection.execute("PRAGMA foreign_keys = ON")
    with connection:
        connection.execute("""
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                margin REAL NOT NULL,
                predictions_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE annotations (
                sample_id INTEGER PRIMARY KEY REFERENCES samples(id),
                category TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute(
            "CREATE INDEX samples_margin_idx ON samples(margin, id)"
        )
        connection.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?)",
            records,
        )

artifacts = ROOT / "artifacts"
artifacts.mkdir(exist_ok=True)
joblib.dump(model, artifacts / "annotation_model.joblib")
(artifacts / "annotation_manifest.json").write_text(
    json.dumps({
        "seed": 7,
        "initial_ids": [int(value) for value in initial_ids],
        "categories": [str(value) for value in model.classes_],
        "model_version": "initial-385-balanced",
    }, indent=2),
    encoding="utf-8",
)

print(f"Bereit: {len(initial_ids)} Trainingsbeispiele.")
print(f"Warteschlange: {len(records)} Texte.")
print(f"Validierung: {len(validation)} getrennte Texte.")
print("Datenbank: data/annotations.sqlite3")

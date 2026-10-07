from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import uuid

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
manifest_path = ROOT / "artifacts" / "annotation_manifest.json"
db_path = ROOT / "data" / "annotations.sqlite3"

if not manifest_path.exists() or not db_path.exists():
    raise SystemExit("Bitte zuerst prepare_annotation.py ausführen.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

with closing(sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)) as connection:
    annotations = pd.read_sql_query("""
        SELECT a.sample_id, s.text, a.category
        FROM annotations AS a
        JOIN samples AS s ON s.id = a.sample_id
        ORDER BY a.sample_id
    """, connection)

if annotations.empty:
    raise SystemExit("Es sind noch keine Annotationen gespeichert.")

df = pd.read_csv(ROOT / "data" / "train.csv")
train, validation = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["category"]
)

initial_ids = manifest["initial_ids"]
annotation_ids = set(annotations["sample_id"])
assert set(initial_ids).issubset(train.index)
assert annotation_ids.issubset(train.index)
assert set(initial_ids).isdisjoint(annotation_ids)
assert (set(initial_ids) | annotation_ids).isdisjoint(validation.index)
assert set(annotations["category"]).issubset(manifest["categories"])

initial = train.loc[initial_ids, ["text", "category"]]
training_data = pd.concat(
    [initial, annotations[["text", "category"]]],
    ignore_index=True,
)

old_model_path = ROOT / "artifacts" / manifest.get(
    "model_file", "annotation_model.joblib"
)
old_model = joblib.load(old_model_path)

before = f1_score(
    validation["category"],
    old_model.predict(validation["text"]),
    average="macro",
)

new_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    )),
])

print(f"Trainiere mit {len(training_data)} Beispielen …", flush=True)
new_model.fit(training_data["text"], training_data["category"])

after = f1_score(
    validation["category"],
    new_model.predict(validation["text"]),
    average="macro",
)

run_id = uuid.uuid4().hex[:12]
run_dir = ROOT / "artifacts" / "annotation_runs" / run_id
run_dir.mkdir(parents=True)
joblib.dump(new_model, run_dir / "model.joblib")

metadata = {
    "run_id": run_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_manifest": manifest,
    "annotations": annotations[["sample_id", "category"]].to_dict("records"),
    "training_examples": len(training_data),
}
(run_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

report = {
    "run_id": run_id,
    "evaluation": "validation",
    "validation_examples": len(validation),
    "human_annotations": len(annotations),
    "training_examples": len(training_data),
    "macro_f1_before": before,
    "macro_f1_after": after,
    "difference": after - before,
}
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
(reports / f"annotation_training_{run_id}.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print(f"Gespeicherte menschliche Labels: {len(annotations)}")
print(f"Macro-F1 vorher: {before:.4f}")
print(f"Macro-F1 danach: {after:.4f}")
print(f"Änderung: {(after - before) * 100:+.2f} F1-Punkte")
print(f"Trainingslauf: {run_id}")
print("Das neue Modell wurde separat gespeichert.")

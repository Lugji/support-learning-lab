import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/train.csv")

# 80 % zum Lernen, 20 % zur Validierung
X_train, X_val, y_train, y_val = train_test_split(
    df["text"],
    df["category"],
    test_size=0.2,
    random_state=42,
    stratify=df["category"],
)

# Text in Zahlen umwandeln und Kategorien lernen
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000)),
])

print("Training startet …", flush=True)
model.fit(X_train, y_train)

# Bewertung auf den zurückgehaltenen Anfragen
predictions = model.predict(X_val
)

print(f"Accuracy: {accuracy_score(y_val, predictions):.3f}")
print(
    f"Macro-F1: "
    f"{f1_score(y_val, predictions, average='macro'):.3f}"
)

# Eine eigene englische Anfrage ausprobieren
request = "My card has not arrived yet."
category = model.predict([request])[0]

print("\nAnfrage:", request)
print("Vorhergesagte Kategorie:", category)

# Die drei höchsten Modellwahrscheinlichkeiten für unsere Anfrage
probabilities = model.predict_proba([request])[0]
top_indices = probabilities.argsort()[-3:][::-1]

print("\nTop-3-Kategorien für die Beispielanfrage:")
for index in top_indices:
    print(
        f"{model.classes_[index]}: "
        f"{probabilities[index]:.3f}"
    )

# Falsche Vorhersagen auf den Validierungsdaten untersuchen
results = pd.DataFrame({
    "text": X_val.to_numpy(),
    "actual": y_val.to_numpy(),
    "predicted": predictions,
})

errors = results[results["actual"] != results["predicted"]]

print("\nAnzahl falscher Vorhersagen:", len(errors))
print("\nAcht zufällig ausgewählte Fehler:")
print(
    errors.sample(
        n=min(8, len(errors)),
        random_state=42,
    ).to_string(index=False)
)

from pathlib import Path
import json
import joblib

Path("artifacts").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# Speichert Textverarbeitung und Klassifikator zusammen
joblib.dump(model, "artifacts/baseline_model.joblib")

metrics = {
    "evaluation_split": "validation",
    "random_state": 42,
    "training_examples": len(X_train),
    "validation_examples": len(X_val),
    "accuracy": float(accuracy_score(y_val, predictions)),
    "macro_f1": float(
        f1_score(y_val, predictions, average="macro")
    ),
    "incorrect_predictions": len(errors),
}

with open("reports/baseline_metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)

errors.to_csv("reports/baseline_errors.csv", index=False)

print("\nModell und Ergebnisse gespeichert.")

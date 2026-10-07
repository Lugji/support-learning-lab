from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def make_model(class_weight=None):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
        )),
    ])


df = pd.read_csv("data/train.csv")
train, validation = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["category"],
)
train = train.reset_index(drop=True)

rng = np.random.default_rng(7)
initial_ids = []

for category in sorted(train["category"].unique()):
    ids = train.index[train["category"] == category].to_numpy()
    initial_ids.extend(rng.choice(ids, size=5, replace=False))

initial_ids = np.array(initial_ids, dtype=int)
pool_ids = np.setdiff1d(train.index.to_numpy(), initial_ids)

# Dasselbe Ausgangsmodell wie im ursprünglichen Experiment.
initial_model = make_model()
initial_model.fit(
    train.loc[initial_ids, "text"],
    train.loc[initial_ids, "category"],
)

random_ids = np.random.default_rng(7).choice(
    pool_ids, size=100, replace=False
)

probabilities = initial_model.predict_proba(
    train.loc[pool_ids, "text"]
)
ordered = np.sort(probabilities, axis=1)
margins = ordered[:, -1] - ordered[:, -2]

shuffled = np.random.default_rng(7).permutation(len(pool_ids))
ranking = shuffled[
    np.argsort(margins[shuffled], kind="stable")
]
margin_ids = pool_ids[ranking[:100]]

datasets = {
    "start": initial_ids,
    "random": np.concatenate([initial_ids, random_ids]),
    "margin": np.concatenate([initial_ids, margin_ids]),
}

results = []

for selection, ids in datasets.items():
    for weight in [None, "balanced"]:
        model = make_model(weight)
        model.fit(
            train.loc[ids, "text"],
            train.loc[ids, "category"],
        )

        predictions = model.predict(validation["text"])
        score = f1_score(
            validation["category"],
            predictions,
            average="macro",
        )

        row = {
            "selection": selection,
            "labels": len(ids),
            "class_weight": "none" if weight is None else weight,
            "macro_f1": score,
            "predicted_categories": len(np.unique(predictions)),
        }
        results.append(row)

        print(
            f"{selection:6s} | Labels: {len(ids)} | "
            f"Gewichtung: {row['class_weight']:8s} | "
            f"Macro-F1: {score:.3f} | "
            f"Vorhergesagte Kategorien: "
            f"{row['predicted_categories']}/77",
            flush=True,
        )

Path("reports").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(
    "reports/class_weight_diagnostic.csv",
    index=False,
)

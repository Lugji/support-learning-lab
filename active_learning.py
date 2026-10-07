from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEEDS = [7, 21, 42]
ROUNDS = 5
BATCH_SIZE = 100

df = pd.read_csv("data/train.csv")

train, validation = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["category"],
)

assert set(train.index).isdisjoint(validation.index)
train = train.reset_index(drop=True)

Path("reports").mkdir(exist_ok=True)
results = []

for seed in SEEDS:
    # Fünf Startbeispiele pro Kategorie.
    initial_rng = np.random.default_rng(seed)
    initial_ids = []

    for category in sorted(train["category"].unique()):
        category_ids = train.index[
            train["category"] == category
        ].to_numpy()

        initial_ids.extend(
            initial_rng.choice(category_ids, size=5, replace=False)
        )

    # Beide Strategien erhalten exakt dieselben Startbeispiele.
    for strategy in ["random", "margin"]:
        labeled_ids = np.array(initial_ids, dtype=int)
        pool_ids = np.setdiff1d(train.index.to_numpy(), labeled_ids)
        selection_rng = np.random.default_rng(seed)

        for round_number in range(ROUNDS + 1):
            model = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(max_iter=1000)),
            ])

            labeled = train.loc[labeled_ids]
           
            counts = labeled["category"].value_counts()

            print(
                f"  Beispiele pro Kategorie: "
                f"Minimum={counts.min()}, "
                f"Maximum={counts.max()}",
                flush=True,
            )

            counts.rename_axis("category").reset_index(
                name="count"
            ).to_csv(
                f"reports/class_counts_{seed}_{strategy}_{round_number}.csv",
                index=False,
            )

            model.fit(labeled["text"], labeled["category"])

            predictions = model.predict(validation["text"])
            score = f1_score(
                validation["category"],
                predictions,
                average="macro",
            )

            results.append({
                "seed": seed,
                "strategy": strategy,
                "round": round_number,
                "labeled_examples": len(labeled_ids),
                "macro_f1": score,
            })

            print(
                f"Seed {seed} | {strategy:6s} | "
                f"Labels: {len(labeled_ids)} | Macro-F1: {score:.3f}",
                flush=True,
            )

            # Zwischenstand nach jeder Auswertung sichern.
            pd.DataFrame(results).to_csv(
                "reports/active_learning_results.csv",
                index=False,
            )

            if round_number == ROUNDS:
                break

            if strategy == "random":
                selected_ids = selection_rng.choice(
                    pool_ids,
                    size=BATCH_SIZE,
                    replace=False,
                )
            else:
                # Auswahl ausschließlich anhand der Texte.
                probabilities = model.predict_proba(
                    train.loc[pool_ids, "text"]
                )
                sorted_probabilities = np.sort(probabilities, axis=1)
                margins = (
                    sorted_probabilities[:, -1]
                    - sorted_probabilities[:, -2]
                )

                # Zufällige Reihenfolge bei gleichen Margins.
                shuffled = selection_rng.permutation(len(pool_ids))
                ranking = shuffled[
                    np.argsort(margins[shuffled], kind="stable")
                ]
                selected_ids = pool_ids[ranking[:BATCH_SIZE]]

            # Erst jetzt werden die ausgewählten Labels verfügbar.
            labeled_ids = np.concatenate([labeled_ids, selected_ids])
            pool_ids = np.setdiff1d(pool_ids, selected_ids)

summary = (
    pd.DataFrame(results)
    .groupby(["strategy", "labeled_examples"])["macro_f1"]
    .agg(["mean", "std"])
    .reset_index()
)

summary.to_csv("reports/active_learning_summary.csv", index=False)

print("\nVergleich über drei Zufallsstarts:")
print(summary.round(3).to_string(index=False))

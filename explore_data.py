import pandas as pd

df = pd.read_csv("data/train.csv")

print("Erste fünf Anfragen:")
print(df.head().to_string(index=False))

print("\nAnzahl der Anfragen:", len(df))
print("Anzahl der Kategorien:", df["category"].nunique())

print("\nFehlende Werte pro Spalte:")
print(df.isna().sum())

print("\nAnzahl der Anfragen pro Kategorie:")
print(df["category"].value_counts())

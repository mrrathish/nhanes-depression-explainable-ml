
import pandas as pd

path = "data/raw/DPQ_L.xpt"
df = pd.read_sas(path, format="xport")

phq_cols = [
    "DPQ010", "DPQ020", "DPQ030",
    "DPQ040", "DPQ050", "DPQ060",
    "DPQ070", "DPQ080", "DPQ090"
]

print("\nTotal DPQ records:", len(df))

print("\nRaw PHQ-9 response distributions:")
for col in phq_cols:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False).sort_index())

print("\nFirst 10 participants:")
print(df[["SEQN"] + phq_cols].head(10).to_string())

# Verify that all PHQ items are 0-3.
clean = df[phq_cols].where(
    df[phq_cols].isin([0, 1, 2, 3])
)

complete = clean.notna().all(axis=1)
scores = clean.sum(axis=1, min_count=9)

print("\nComplete PHQ-9 questionnaires:", complete.sum())

print("\nPHQ-9 score distribution:")
print(scores.value_counts(dropna=False).sort_index())

print("\nPHQ-9 >= 10:", (scores >= 10).sum())
print("PHQ-9 < 10:", (scores < 10).sum())

print("\nPHQ-9 summary:")
print(scores.describe())

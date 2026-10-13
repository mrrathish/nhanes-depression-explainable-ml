
from pathlib import Path
import json
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
REPORTS = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

FILES = ["DEMO", "DPQ", "SLQ", "PAQ", "BMX", "SMQ"]
PHQ = [f"DPQ{i:03d}" for i in range(10, 100, 10)]

COLUMNS = {
    "DEMO": [
        "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
        "DMDEDUC2", "INDFMPIR", "WTINT2YR",
        "WTMEC2YR", "SDMVSTRA", "SDMVPSU"
    ],
    "DPQ": ["SEQN"] + PHQ,
    "SLQ": ["SEQN", "SLD012", "SLD013"],
    "PAQ": ["SEQN", "PAD680"],
    "BMX": ["SEQN", "BMXBMI"],
    "SMQ": ["SEQN", "SMQ020"]
}

tables = {}

for name in FILES:
    path = RAW / f"{name}_L.xpt"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_sas(path, format="xport")

    # Correct near-zero values produced by SAS XPT decoding.
    numeric_cols = df.select_dtypes(include="number").columns

    df[numeric_cols] = df[numeric_cols].mask(
        df[numeric_cols].abs() < 1e-50,
        0.0
    )

    missing = set(COLUMNS[name]) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing {sorted(missing)}")

    if df.SEQN.isna().any() or df.SEQN.duplicated().any():
        raise ValueError(f"{name}: invalid SEQN identifiers")

    tables[name] = df[COLUMNS[name]].copy()
    print(f"{name}: {len(df):,} records")

# Retain demographic participants, then join component data.
data = tables["DEMO"]

for name in FILES[1:]:
    data = data.merge(
        tables[name],
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

# Define adult population.
adults = data.loc[data.RIDAGEYR >= 20].copy()

# PHQ-9: only responses 0, 1, 2, 3 are valid.
for col in PHQ:
    adults[col] = adults[col].where(
        adults[col].isin([0, 1, 2, 3])
    )

adults["phq9_total"] = adults[PHQ].sum(
    axis=1, min_count=9
)

adults["elevated_symptoms"] = np.where(
    adults.phq9_total.isna(),
    np.nan,
    (adults.phq9_total >= 10).astype(int)
)

# Clean selected predictor fields.
adults["SLD012"] = adults["SLD012"].where(
    adults["SLD012"].between(2, 12)
)
adults["SLD013"] = adults["SLD013"].where(
    adults["SLD013"].between(2, 12)
)

adults["PAD680"] = adults["PAD680"].where(
    adults["PAD680"].between(0, 1320)
)
adults["sedentary_hours"] = adults.PAD680 / 60

adults["SMQ020"] = adults["SMQ020"].where(
    adults["SMQ020"].isin([1, 2])
)

adults["DMDEDUC2"] = adults["DMDEDUC2"].where(
    adults["DMDEDUC2"].isin([1, 2, 3, 4, 5])
)

adults["INDFMPIR"] = adults["INDFMPIR"].where(
    adults["INDFMPIR"].between(0, 5)
)

adults["BMXBMI"] = adults["BMXBMI"].where(
    adults["BMXBMI"] > 0
)

# Restrict outcome statistics to complete PHQ-9 responses.
valid = adults.dropna(subset=["elevated_symptoms"])

# Basic weighted point estimate, not a design-based CI.
weighted = valid.loc[
    valid.WTINT2YR.notna() & (valid.WTINT2YR > 0)
]

weighted_prevalence = (
    np.average(
        weighted.elevated_symptoms,
        weights=weighted.WTINT2YR
    )
    if len(weighted) else None
)

predictors = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3",
    "DMDEDUC2", "INDFMPIR", "SLD012",
    "sedentary_hours", "BMXBMI", "SMQ020"
]

report = {
    "demographic_records": len(data),
    "adults_20_plus": len(adults),
    "valid_phq9": len(valid),
    "positive_cases": int(valid.elevated_symptoms.sum()),
    "negative_cases": int(
        (valid.elevated_symptoms == 0).sum()
    ),
    "unweighted_prevalence": (
        float(valid.elevated_symptoms.mean())
        if len(valid) else None
    ),
    "interview_weighted_prevalence": (
        float(weighted_prevalence)
        if weighted_prevalence is not None else None
    ),
    "complete_predictor_cases": int(
        valid[predictors].notna().all(axis=1).sum()
    ),
    "missing_percent": (
        adults[predictors]
        .isna().mean().mul(100).round(2).to_dict()
    )
}

adults.to_parquet(OUT / "nhanes_adults.parquet", index=False)

with open(REPORTS / "data_quality.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))

# Analyze participants with valid PHQ-9 outcomes.
outcome_valid = adults.dropna(
    subset=["elevated_symptoms"]
).copy()

# Complete cases for all selected predictors.
complete = outcome_valid.dropna(
    subset=predictors
).copy()

print("\n=== MODEL FEASIBILITY REPORT ===")

print("Valid PHQ-9 participants:", len(outcome_valid))
print("Complete predictor cases:", len(complete))

positive = int(complete["elevated_symptoms"].sum())
negative = len(complete) - positive

print("Complete-case positive:", positive)
print("Complete-case negative:", negative)

print(
    "Complete-case prevalence:",
    round(100 * positive / len(complete), 2)
    if len(complete) else None
)

print("\nPositive cases by variable availability:")

for variable in predictors:
    subset = outcome_valid.dropna(subset=[variable])
    print(
        variable,
        "N =", len(subset),
        "Positive =", int(subset["elevated_symptoms"].sum())
    )

print("\nSurvey design diagnostics:")
print(
    "Strata:",
    outcome_valid["SDMVSTRA"].nunique()
)
print(
    "PSUs:",
    outcome_valid[
        ["SDMVSTRA", "SDMVPSU"]
    ].drop_duplicates().shape[0]
)

print("\nPHQ-9 score distribution:")
print(
    outcome_valid["phq9_total"]
    .describe()
)


feasibility = {
    "valid_phq9": len(outcome_valid),
    "complete_cases": len(complete),
    "complete_case_positive": positive,
    "complete_case_negative": negative,
    "complete_case_prevalence": (
        positive / len(complete) if len(complete) else None
    ),
    "retention_rate": (
        len(complete) / len(outcome_valid)
        if len(outcome_valid) else None
    ),
    "predictor_missingness_outcome_valid": (
        outcome_valid[predictors]
        .isna().mean().mul(100).round(2).to_dict()
    ),
    "strata": outcome_valid["SDMVSTRA"].nunique(),
    "psus": outcome_valid[
        ["SDMVSTRA", "SDMVPSU"]
    ].drop_duplicates().shape[0]
}

with open(REPORTS / "model_feasibility.json", "w") as f:
    json.dump(feasibility, f, indent=2)


print("\n=== OUTCOME-VALID MISSINGNESS ===")

missing = (
    outcome_valid[predictors]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

print(missing)

print("\n=== COMPLETE-CASE RETENTION ===")

print(
    f"Retained: {len(complete):,} / "
    f"{len(outcome_valid):,}"
)

print(
    f"Excluded: "
    f"{len(outcome_valid) - len(complete):,}"
)

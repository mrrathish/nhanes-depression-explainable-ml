"""Create fixed, leakage-free, stratified NHANES train/test partitions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_common import ROOT, DATA, OUT, FEATURES, NUMERIC, CATEGORICAL, META, SEED, dirs, dump_json


def main():
    dirs()
    source = DATA / 'nhanes_adults.parquet'
    df = pd.read_parquet(source)
    needed = ['SEQN', 'elevated_symptoms', 'phq9_total'] + FEATURES
    missing = sorted(set(needed) - set(df.columns))
    if missing: raise ValueError(f'Missing expected columns: {missing}')
    if df.SEQN.isna().any() or df.SEQN.duplicated().any():
        raise ValueError('SEQN must be unique and nonmissing')
    # The derived outcome is only valid for participants with complete, valid PHQ-9.
    phq = pd.to_numeric(df['phq9_total'], errors='coerce')
    phq = phq.mask(phq.abs() < 1e-50, 0)
    eligible = phq.between(0, 27)
    df = df.loc[eligible].copy()
    expected = (phq.loc[eligible] >= 10).astype(int)
    actual = pd.to_numeric(df['elevated_symptoms'], errors='coerce')
    actual = actual.mask(actual.abs() < 1e-50, 0)
    if not actual.isin([0, 1]).all() or not (actual.astype(int).to_numpy() == expected.to_numpy()).all():
        raise ValueError('Outcome inconsistent with PHQ-9 >= 10; check source derivation')
    df['elevated_symptoms'] = expected
    for col in NUMERIC + CATEGORICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].mask(df[col].abs() < 1e-50, 0)
    # NHANES questionnaire missing codes: do not impute refusal/don't know as real categories.
    valid_cat = {'RIAGENDR': [1, 2], 'RIDRETH3': [1, 2, 3, 4, 6, 7],
                 'DMDEDUC2': [1, 2, 3, 4, 5], 'SMQ020': [1, 2]}
    for col, allowed in valid_cat.items():
        df.loc[~df[col].isin(allowed), col] = np.nan
        df[col] = df[col].map(lambda v: str(int(v)) if pd.notna(v) else np.nan).astype(object)  # nominal categories
    valid_ranges = {'RIDAGEYR': (20, 120), 'INDFMPIR': (0, 5),
                    'SLD012': (2, 12), 'sedentary_hours': (0, 24), 'BMXBMI': (0, 100)}
    for col, (lo, hi) in valid_ranges.items():
        df.loc[~df[col].between(lo, hi, inclusive='both'), col] = np.nan
    if len(df) < 100 or df.elevated_symptoms.nunique() != 2:
        raise ValueError('Insufficient eligible data or missing outcome class')
    keep = FEATURES + ['elevated_symptoms'] + [c for c in META if c in df.columns]
    df = df[keep].copy()
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.2,
                                            stratify=df.elevated_symptoms, random_state=SEED)
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    train.to_parquet(DATA / 'ml_train.parquet', index=False)
    test.to_parquet(DATA / 'ml_test.parquet', index=False)
    miss = pd.DataFrame({'variable': FEATURES,
                         'missing_n': [int(df[c].isna().sum()) for c in FEATURES],
                         'missing_pct': [round(100*df[c].isna().mean(), 2) for c in FEATURES]})
    miss.to_csv(OUT / 'predictor_missingness.csv', index=False)
    summary = {'eligible_n': len(df), 'positive_n': int(df.elevated_symptoms.sum()),
               'train_n': len(train), 'train_positive': int(train.elevated_symptoms.sum()),
               'test_n': len(test), 'test_positive': int(test.elevated_symptoms.sum()),
               'random_seed': SEED, 'test_fraction': .2,
               'note': 'Individual-level stratified split; not PSU-grouped or design-weighted.'}
    dump_json(OUT / 'split_summary.json', summary)
    print(summary)
    print('Saved ml_train.parquet, ml_test.parquet and predictor_missingness.csv')


if __name__ == '__main__': main()

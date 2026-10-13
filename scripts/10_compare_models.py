"""Paired participant bootstrap for AUC comparisons; descriptive survey-weighted AUC sensitivity."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ml_common import dirs, OUT, SEED

NAMES = ('logistic_regression', 'random_forest', 'xgboost')
B = 2000


def boot_indices(y, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate((rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)))


def main():
    dirs()
    df = pd.read_csv(OUT / 'test_predictions.csv')
    y = df.y_true.to_numpy(dtype=int)
    if len(np.unique(y)) != 2: raise ValueError('Both classes required for ROC-AUC')
    rng = np.random.default_rng(SEED)
    indices = [boot_indices(y, rng) for _ in range(B)]
    estimates = {}
    rows = []
    for name in NAMES:
        p = df[name].to_numpy()
        auc = roc_auc_score(y, p)
        draws = np.array([roc_auc_score(y[idx], p[idx]) for idx in indices])
        estimates[name] = draws
        lo, hi = np.quantile(draws, [.025, .975])
        row = {'model': name, 'auc': auc, 'ci_lower': lo, 'ci_upper': hi,
               'ci_method': 'individual_stratified_bootstrap'}
        if 'WTMEC2YR' in df:
            w = df.WTMEC2YR.to_numpy(dtype=float)
            valid = np.isfinite(w) & (w > 0)
            if valid.sum() and len(np.unique(y[valid])) == 2:
                row['mec_weighted_auc_point_estimate'] = roc_auc_score(
                    y[valid], p[valid], sample_weight=w[valid])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'auc_confidence_intervals.csv', index=False)
    comparisons = []
    for name in ('random_forest', 'xgboost'):
        difference = estimates[name] - estimates['logistic_regression']
        lo, hi = np.quantile(difference, [.025, .975])
        comparisons.append({'comparison': f'{name} minus logistic_regression',
                            'auc_difference': roc_auc_score(y, df[name]) - roc_auc_score(y, df['logistic_regression']),
                            'ci_lower': lo, 'ci_upper': hi,
                            'ci_excludes_zero': bool(lo > 0 or hi < 0)})
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_auc_comparisons.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print('CAUTION: bootstrap CIs assume independent individuals; not NHANES design-based CIs.')


if __name__ == '__main__': main()

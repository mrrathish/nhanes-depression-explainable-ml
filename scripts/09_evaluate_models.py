"""One-time held-out test evaluation. Requires completed tuning."""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from ml_common import dirs, read_split, model_path, OUT, FIG, metric_summary

NAMES = ('logistic_regression', 'random_forest', 'xgboost')


def main():
    dirs()
    if not (OUT / 'tuning_complete.txt').exists():
        raise RuntimeError('Run 08_cross_validation.py before evaluating the test set')
    X, y, meta = read_split('test')
    predictions = pd.DataFrame({'SEQN': meta.SEQN if 'SEQN' in meta else np.arange(len(y)),
                                'y_true': y.to_numpy()})
    for col in ('WTMEC2YR', 'SDMVSTRA', 'SDMVPSU'):
        if col in meta: predictions[col] = meta[col].to_numpy()
    metrics = []
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))
    fig_cal, ax_cal = plt.subplots(figsize=(7, 5.5))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5.5))
    for name in NAMES:
        model = joblib.load(model_path(name))
        prob = model.predict_proba(X)[:, 1]
        predictions[name] = prob
        met = metric_summary(y, prob)
        met['model'] = name
        metrics.append(met)
        fpr, tpr, _ = roc_curve(y, prob)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC={met["roc_auc"]:.3f})')
        p, r, _ = precision_recall_curve(y, prob)
        ax_pr.plot(r, p, label=f'{name} (AP={met["average_precision"]:.3f})')
        observed, predicted = calibration_curve(y, prob, n_bins=8, strategy='quantile')
        ax_cal.plot(predicted, observed, marker='o', label=name)
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT / 'test_metrics.csv', index=False)
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_roc.set(xlabel='False positive rate', ylabel='True positive rate', title='Held-out ROC curves')
    ax_roc.legend(fontsize=8)
    ax_pr.axhline(y.mean(), linestyle='--', color='gray', label='Test prevalence')
    ax_pr.set(xlabel='Recall', ylabel='Precision', title='Held-out precision-recall curves')
    ax_pr.legend(fontsize=8)
    ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_cal.set(xlabel='Mean predicted probability', ylabel='Observed fraction', title='Calibration (unweighted)')
    ax_cal.legend(fontsize=8)
    for fig, filename in ((fig_roc, 'roc_curves.png'), (fig_pr, 'pr_curves.png'),
                          (fig_cal, 'calibration_curves.png')):
        fig.tight_layout()
        fig.savefig(FIG / filename, dpi=180)
        plt.close(fig)
    print(pd.DataFrame(metrics)[['model', 'n', 'positives', 'roc_auc', 'average_precision', 'brier']].to_string(index=False))
    print('Saved held-out predictions, metrics, and three figures')


if __name__ == '__main__': main()

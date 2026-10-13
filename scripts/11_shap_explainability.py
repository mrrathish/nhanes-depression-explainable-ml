"""SHAP explanations for the tuned XGBoost model (or Random Forest fallback)."""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_common import dirs, read_split, model_path, OUT, FIG, SEED


def class_one_values(explanation):
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base.ndim == 2: base = base[:, 1]
    return values, base


def main():
    import shap
    dirs()
    X, y, meta = read_split('test')
    name = 'xgboost'
    if not model_path(name).exists(): name = 'random_forest'
    pipe = joblib.load(model_path(name))
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(X), size=min(300, len(X)), replace=False))
    subset = X.iloc[chosen]
    transformed = pipe.named_steps['preprocess'].transform(subset)
    names = pipe.named_steps['preprocess'].get_feature_names_out()
    transformed = np.asarray(transformed, dtype=float)
    model = pipe.named_steps['model']
    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)
    values, base = class_one_values(explanation)
    if values.shape != transformed.shape:
        raise ValueError(f'Unexpected SHAP shape: {values.shape}, expected {transformed.shape}')
    importance = pd.DataFrame({'feature': names,
                'mean_absolute_shap': np.abs(values).mean(axis=0)}).sort_values('mean_absolute_shap', ascending=False)
    importance.to_csv(OUT / 'shap_feature_importance.csv', index=False)
    pd.DataFrame(values, columns=names).to_csv(OUT / 'shap_values_sample.csv', index=False)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=18)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_beeswarm.png', dpi=180, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(9, 6))
    shap.summary_plot(values, transformed, feature_names=names, plot_type='bar', show=False, max_display=18)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_importance.png', dpi=180, bbox_inches='tight')
    plt.close()
    # A waterfall plot explains the first reproducibly sampled test observation.
    first = shap.Explanation(values=values[0], base_values=np.asarray(base).reshape(-1)[0],
                             data=transformed[0], feature_names=list(names))
    shap.plots.waterfall(first, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_waterfall.png', dpi=180, bbox_inches='tight')
    plt.close()
    # Dependence-style scatter for the highest-ranking original numeric feature.
    for original in ('SLD012', 'sedentary_hours'):
        idx = next((i for i, n in enumerate(names) if n == f'num__{original}'), None)
        if idx is None: continue
        plt.figure(figsize=(7, 5))
        plt.scatter(pd.to_numeric(subset[original], errors='coerce'), values[:, idx], alpha=.5, s=14)
        plt.xlabel(original + ' (original units)')
        plt.ylabel('SHAP contribution (model raw output)')
        plt.title(f'{original} SHAP dependence')
        plt.tight_layout()
        plt.savefig(FIG / f'shap_dependence_{original}.png', dpi=180)
        plt.close()
    print(f'Saved SHAP outputs for {name}; {len(chosen)} test observations.')
    print('SHAP scale depends on model/explainer; do not interpret as causal effects.')


if __name__ == '__main__': main()

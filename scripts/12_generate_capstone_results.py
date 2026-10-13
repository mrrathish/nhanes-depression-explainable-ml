"""Aggregate computed ML results into an audit-friendly capstone summary."""
import pandas as pd
from ml_common import dirs, OUT, FIG


def main():
    dirs()
    needed = ['cross_validation_summary.csv', 'test_metrics.csv',
              'auc_confidence_intervals.csv', 'paired_auc_comparisons.csv']
    absent = [f for f in needed if not (OUT / f).exists()]
    if absent: raise FileNotFoundError('Run scripts 08-10 first: ' + ', '.join(absent))
    cv = pd.read_csv(OUT / needed[0])
    metrics = pd.read_csv(OUT / needed[1])
    auc = pd.read_csv(OUT / needed[2])
    diffs = pd.read_csv(OUT / needed[3])
    summary = metrics.merge(auc[['model', 'ci_lower', 'ci_upper']], on='model').merge(
        cv[['model', 'cv_roc_auc_mean', 'cv_roc_auc_sd']], on='model')
    summary.to_csv(OUT / 'capstone_model_table.csv', index=False)
    lines = [
        '# RQ3 / RQ4: Predictive Modeling Results', '',
        'Outcome: PHQ-9 >= 10 (elevated depressive symptoms; not clinical diagnosis).',
        'Development: fixed stratified 80% training split; 5-fold CV for tuning.',
        'Evaluation: fixed untouched 20% held-out test split.', '',
        '## Model results', '', summary.round(4).to_markdown(index=False), '',
        '## Paired AUC differences', '', diffs.round(4).to_markdown(index=False), '',
        '## Interpretation cautions', '',
        '- Associations and SHAP attributions are not causal effects.',
        '- Main test-set estimates are unweighted and based on an individual-level split.',
        '- Bootstrap confidence intervals are not NHANES design-based uncertainty intervals.',
        '- Survey-weighted AUC point estimates, where present, are descriptive sensitivity checks.',
        '- The outcome is current elevated symptoms, not prospective risk or a clinical diagnosis.',
        '- Compare AUC differences with their confidence intervals; avoid claims based only on ranking.',
        '- Validate the NHANES 2021-2023 cycle-specific weighting rules before submission.', '',
        '## Figures', '',
    ]
    for name in ('roc_curves.png', 'pr_curves.png', 'calibration_curves.png',
                 'shap_beeswarm.png', 'shap_importance.png', 'shap_waterfall.png',
                 'shap_dependence_SLD012.png', 'shap_dependence_sedentary_hours.png'):
        if (FIG / name).exists(): lines.append(f'- reports/figures/{name}')
    (OUT / 'capstone_results.md').write_text('\n'.join(lines) + '\n')
    print('Saved', OUT / 'capstone_results.md')
    print('Saved', OUT / 'capstone_model_table.csv')


if __name__ == '__main__': main()

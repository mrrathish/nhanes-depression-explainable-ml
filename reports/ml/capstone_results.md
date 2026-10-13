# RQ3 / RQ4: Predictive Modeling Results

Outcome: PHQ-9 >= 10 (elevated depressive symptoms; not clinical diagnosis).
Development: fixed stratified 80% training split; 5-fold CV for tuning.
Evaluation: fixed untouched 20% held-out test split.

## Model results

|    n |   positives |   roc_auc |   average_precision |   brier |   sensitivity |   specificity |   precision |     f1 |   tp |   fp |   tn |   fn |   threshold | model               |   ci_lower |   ci_upper |   cv_roc_auc_mean |   cv_roc_auc_sd |
|-----:|------------:|----------:|--------------------:|--------:|--------------:|--------------:|------------:|-------:|-----:|-----:|-----:|-----:|------------:|:--------------------|-----------:|-----------:|------------------:|----------------:|
| 1042 |         132 |    0.6698 |              0.2339 |  0.2164 |        0.5758 |        0.667  |      0.2005 | 0.2975 |   76 |  303 |  607 |   56 |         0.5 | logistic_regression |     0.6187 |     0.7175 |            0.7094 |          0.0302 |
| 1042 |         132 |    0.6814 |              0.2552 |  0.1049 |        0      |        1      |      0      | 0      |    0 |    0 |  910 |  132 |         0.5 | random_forest       |     0.6334 |     0.7294 |            0.7241 |          0.0238 |
| 1042 |         132 |    0.684  |              0.2568 |  0.1051 |        0.0227 |        0.9978 |      0.6    | 0.0438 |    3 |    2 |  908 |  129 |         0.5 | xgboost             |     0.634  |     0.7322 |            0.7163 |          0.0271 |

## Paired AUC differences

| comparison                              |   auc_difference |   ci_lower |   ci_upper | ci_excludes_zero   |
|:----------------------------------------|-----------------:|-----------:|-----------:|:-------------------|
| random_forest minus logistic_regression |           0.0115 |    -0.012  |     0.0359 | False              |
| xgboost minus logistic_regression       |           0.0141 |    -0.0087 |     0.0379 | False              |

## Interpretation cautions

- Associations and SHAP attributions are not causal effects.
- Main test-set estimates are unweighted and based on an individual-level split.
- Bootstrap confidence intervals are not NHANES design-based uncertainty intervals.
- Survey-weighted AUC point estimates, where present, are descriptive sensitivity checks.
- The outcome is current elevated symptoms, not prospective risk or a clinical diagnosis.
- Compare AUC differences with their confidence intervals; avoid claims based only on ranking.
- Validate the NHANES 2021-2023 cycle-specific weighting rules before submission.

## Figures

- reports/figures/roc_curves.png
- reports/figures/pr_curves.png
- reports/figures/calibration_curves.png
- reports/figures/shap_beeswarm.png
- reports/figures/shap_importance.png
- reports/figures/shap_waterfall.png
- reports/figures/shap_dependence_SLD012.png
- reports/figures/shap_dependence_sedentary_hours.png

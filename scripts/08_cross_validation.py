"""Development-only hyperparameter tuning; save final refitted pipelines."""
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from ml_common import dirs, read_split, make_pipeline, model_path, OUT, dump_json, SEED

PARAMS = {
    'logistic_regression': {'model__C': [.01, .1, 1., 10.],
                            'model__class_weight': [None, 'balanced']},
    'random_forest': {'model__n_estimators': [150, 250, 400],
                      'model__max_depth': [None, 4, 8, 12],
                      'model__min_samples_leaf': [3, 5, 10],
                      'model__class_weight': [None, 'balanced']},
    'xgboost': {'model__n_estimators': [150, 250, 400],
                'model__max_depth': [2, 3, 4],
                'model__learning_rate': [.03, .05, .1],
                'model__subsample': [.8, 1.],
                'model__reg_lambda': [1., 5., 10.]},
}


def main():
    dirs()
    X, y, _ = read_split('train')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, space in PARAMS.items():
        print(f'Tuning {name}...', flush=True)
        search = RandomizedSearchCV(make_pipeline(name), space,
                 n_iter=8 if name == 'logistic_regression' else 12,
                 scoring={'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
                          'neg_brier': 'neg_brier_score'},
                 refit='roc_auc', cv=cv, n_jobs=1, random_state=SEED,
                 return_train_score=False, error_score='raise')
        search.fit(X, y)
        joblib.dump(search.best_estimator_, model_path(name))
        i = search.best_index_
        row = {'model': name, 'cv_roc_auc_mean': search.cv_results_['mean_test_roc_auc'][i],
               'cv_roc_auc_sd': search.cv_results_['std_test_roc_auc'][i],
               'cv_average_precision': search.cv_results_['mean_test_average_precision'][i],
               'cv_brier': -search.cv_results_['mean_test_neg_brier'][i],
               'best_params': str(search.best_params_)}
        rows.append(row)
        pd.DataFrame(search.cv_results_).to_csv(OUT / f'cv_grid_{name}.csv', index=False)
        dump_json(OUT / f'best_params_{name}.json', search.best_params_)
        print(name, 'CV AUC:', round(row['cv_roc_auc_mean'], 4), 'saved', model_path(name))
    pd.DataFrame(rows).to_csv(OUT / 'cross_validation_summary.csv', index=False)
    (OUT / 'tuning_complete.txt').write_text('All three models tuned and refitted on training data.\n')


if __name__ == '__main__': main()

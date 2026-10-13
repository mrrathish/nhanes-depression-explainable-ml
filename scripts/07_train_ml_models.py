"""Fit preliminary Random Forest and XGBoost models, without touching test data."""
import joblib
from ml_common import dirs, read_split, make_pipeline, model_path


def main():
    dirs()
    X, y, _ = read_split('train')
    for name in ('random_forest', 'xgboost'):
        model = make_pipeline(name)
        model.fit(X, y)
        joblib.dump(model, model_path(name))
        print('Saved', model_path(name))
    print('Run 08_cross_validation.py to tune/refit before final test evaluation.')


if __name__ == '__main__': main()

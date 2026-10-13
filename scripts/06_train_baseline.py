"""Fit a prespecified unweighted logistic regression baseline on training records."""
import joblib
from ml_common import dirs, read_split, make_pipeline, model_path, OUT, dump_json


def main():
    dirs()
    X, y, _ = read_split('train')
    pipe = make_pipeline('logistic_regression')
    pipe.fit(X, y)
    joblib.dump(pipe, model_path('logistic_regression'))
    dump_json(OUT / 'baseline_info.json', {'model': 'logistic_regression',
              'training_n': len(y), 'events': int(y.sum()),
              'note': 'Preliminary model; script 08 replaces it with tuned development-only model.'})
    print('Saved', model_path('logistic_regression'))


if __name__ == '__main__': main()

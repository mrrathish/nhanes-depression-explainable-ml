"""Shared, reproducible NHANES depression ML helpers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'reports' / 'ml'
FIG = ROOT / 'reports' / 'figures'
MODELS = ROOT / 'models'
NUMERIC = ['RIDAGEYR', 'INDFMPIR', 'SLD012', 'sedentary_hours', 'BMXBMI']
CATEGORICAL = ['RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'SMQ020']
FEATURES = NUMERIC + CATEGORICAL
META = ['SEQN', 'SDMVSTRA', 'SDMVPSU', 'WTMEC2YR', 'WTINT2YR']
SEED = 42


def dirs():
    for p in (DATA, OUT, FIG, MODELS): p.mkdir(parents=True, exist_ok=True)


def read_split(which):
    path = DATA / f'ml_{which}.parquet'
    if not path.exists(): raise FileNotFoundError(f'{path} missing: run 05_prepare_ml_dataset.py first')
    df = pd.read_parquet(path)
    return df[FEATURES], df['elevated_symptoms'].astype(int), df


def preprocessor():
    num = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=False)),
                    ('scale', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', num, NUMERIC), ('cat', cat, CATEGORICAL)],
                             verbose_feature_names_out=True)


def estimator(name):
    if name == 'logistic_regression':
        return LogisticRegression(max_iter=2500, random_state=SEED)
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
                                      random_state=SEED, n_jobs=-1)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.05,
                             subsample=.85, colsample_bytree=.9,
                             eval_metric='logloss', random_state=SEED,
                             n_jobs=2)
    raise ValueError(name)


def make_pipeline(name):
    return Pipeline([('preprocess', preprocessor()), ('model', estimator(name))])


def model_path(name): return MODELS / f'{name}.joblib'


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def metric_summary(y, prob, threshold=.5):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), positives=int(y.sum()), roc_auc=float(roc_auc_score(y, prob)),
                average_precision=float(average_precision_score(y, prob)),
                brier=float(brier_score_loss(y, prob)),
                sensitivity=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn/(tn+fp)) if tn+fp else np.nan,
                precision=float(precision_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), threshold=threshold)


def positive_weights(df):
    w = pd.to_numeric(df.get('WTMEC2YR'), errors='coerce')
    return w if w is not None else pd.Series(np.nan, index=df.index)

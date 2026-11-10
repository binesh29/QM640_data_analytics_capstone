"""SLA breach risk model: shared feature engineering, training and scoring.

The feature definitions are identical to notebook 04 (only attributes known when a ticket is logged).
Used by src/build_database.py, src/train_model.py, src/score_batch.py, app/api.py and app/dashboard.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'data' / 'db' / 'itsm.sqlite'
MODEL_PATH = ROOT / 'models' / 'sla_breach_rf.joblib'

CAT = ['log_priority', 'log_impact', 'log_urgency', 'log_category', 'log_subcategory', 'log_u_symptom',
       'log_contact_type', 'log_location', 'log_assignment_group', 'log_notify']
CAT_ALL = CAT + ['dow']
NUM = ['caller_freq', 'hour_sin', 'hour_cos', 'weekend', 'log_u_priority_confirmation', 'cmdb_ci_missing']
FEATS = CAT_ALL + NUM
# Raw at-logging fields a caller (ServiceNow, dashboard or API) must supply for one ticket
INPUT_FIELDS = CAT + ['log_caller_id', 'log_cmdb_ci', 'log_u_priority_confirmation', 'opened_at']


def make_features(df, caller_freq):
    """Turn at-logging fields into the 17 model features. caller_freq: dict caller_id -> training-period count."""
    d = df.copy()
    d['opened_at'] = pd.to_datetime(d['opened_at'])
    for c in CAT:
        d[c] = d[c].fillna('Missing').astype(str)
    d['log_u_priority_confirmation'] = d['log_u_priority_confirmation'].astype(str).str.lower().isin(['true', '1']).astype(int)
    d['cmdb_ci_missing'] = d['log_cmdb_ci'].isna().astype(int)
    hour, dow = d['opened_at'].dt.hour, d['opened_at'].dt.dayofweek
    d['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    d['weekend'] = (dow >= 5).astype(int)
    d['dow'] = dow.astype(str)
    d['caller_freq'] = np.log1p(d['log_caller_id'].map(caller_freq).fillna(0))
    return d[FEATS]


def build_pipeline(params, seed=42):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30, sparse_output=True), CAT_ALL),
        ('num', 'passthrough', NUM)])
    rf = RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample', n_jobs=-1, random_state=seed,
                                min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'])
    return Pipeline([('p', pre), ('m', rf)])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class RiskModel:
    """Random forest + Platt recalibration + capacity thresholds, saved as one object."""

    def __init__(self, pipe, platt, caller_freq, thresholds, meta):
        self.pipe, self.platt, self.caller_freq, self.thresholds, self.meta = pipe, platt, caller_freq, thresholds, meta

    def raw_score(self, df):
        return self.pipe.predict_proba(make_features(df, self.caller_freq))[:, 1]

    def predict(self, df):
        """Return a frame with breach probability (calibrated), risk band and suggested action."""
        raw = self.raw_score(df)
        prob = self.platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        band = np.where(raw >= self.thresholds['high'], 'High', np.where(raw >= self.thresholds['medium'], 'Medium', 'Low'))
        action = pd.Series(band).map({'High': 'Review now: confirm assignment group, assign senior resolver, notify duty manager',
                                      'Medium': 'Watch: check at next queue review',
                                      'Low': 'Standard handling'}).values
        return pd.DataFrame({'breach_probability': prob.round(3), 'risk_score_raw': raw.round(3), 'risk_band': band,
                             'suggested_action': action}, index=df.index)


def load_model(path=MODEL_PATH):
    import joblib
    return joblib.load(path)


def model_card(path=MODEL_PATH):
    m = load_model(path)
    return json.dumps(m.meta, indent=1)

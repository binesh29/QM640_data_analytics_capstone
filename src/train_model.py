"""Train and save the deployable SLA breach risk model (models/sla_breach_rf.joblib).

Same data, features, split and hyperparameters as notebooks 04 and 05:
  * random forest fitted on the first 80% of the training period,
  * Platt recalibration fitted on the last 20% of the training period (most recent weeks before the cut-off),
  * risk bands set from the recalibration window: High = top 20% of raw scores, Medium = next 20%.
Reads incidents from the SQLite database (run src/build_database.py first).
Run from the repository root:  python src/train_model.py
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sla_model
from sla_model import ROOT, DB_PATH, MODEL_PATH, RiskModel, build_pipeline, make_features, _logit


def train(db_path=DB_PATH, model_path=MODEL_PATH):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    con = sqlite3.connect(db_path)
    inc = pd.read_sql('SELECT * FROM incidents', con, parse_dates=['opened_at'])
    con.close()
    inc = inc.sort_values('opened_at').reset_index(drop=True)
    cut = int(len(inc) * 0.70)
    tr, te = inc.iloc[:cut], inc.iloc[cut:]
    c2 = int(len(tr) * 0.80)
    fit, cal = tr.iloc[:c2], tr.iloc[c2:]
    caller_freq = fit['log_caller_id'].value_counts().to_dict()
    params = json.load(open(ROOT / 'results' / 'rq4.json'))['cv']['Random forest']['best_params']
    pipe = build_pipeline(params).fit(make_features(fit, caller_freq), fit['sla_breach'])
    raw_cal = pipe.predict_proba(make_features(cal, caller_freq))[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_cal).reshape(-1, 1), cal['sla_breach'])
    thresholds = {'high': float(np.quantile(raw_cal, 0.80)), 'medium': float(np.quantile(raw_cal, 0.60))}
    meta = {'model_version': 'rf-v1.0', 'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'algorithm': 'RandomForestClassifier (400 trees) + Platt recalibration', 'hyperparameters': params,
            'fit_window': [str(fit['opened_at'].min()), str(fit['opened_at'].max())], 'n_fit': len(fit),
            'calibration_window': [str(cal['opened_at'].min()), str(cal['opened_at'].max())], 'n_calibration': len(cal),
            'thresholds_raw': thresholds, 'features': sla_model.FEATS, 'input_fields': sla_model.INPUT_FIELDS}
    model = RiskModel(pipe, platt, caller_freq, thresholds, meta)
    out = model.predict(te)
    meta['holdout'] = {'n': len(te), 'auc': round(roc_auc_score(te['sla_breach'], out['risk_score_raw']), 3),
                       'mean_probability': round(out['breach_probability'].mean(), 3), 'observed_rate': round(te['sla_breach'].mean(), 3),
                       'share_high': round((out['risk_band'] == 'High').mean(), 3),
                       'breach_rate_by_band': te.groupby(out['risk_band'].values)['sla_breach'].mean().round(3).to_dict()}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    json.dump(meta, open(ROOT / 'results' / 'model_card.json', 'w'), indent=1)
    return meta


if __name__ == '__main__':
    print(json.dumps(train(), indent=1))

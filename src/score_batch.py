"""Batch scoring: score incidents in a date window and write the results to the risk_scores table.

In production this would run every few minutes on newly logged tickets; here it replays a window of the event log.
Run from the repository root, e.g.:
    python src/score_batch.py --start 2016-05-02 --end 2016-05-03
"""
import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sla_model import DB_PATH, load_model


def score_window(start, end, db_path=DB_PATH):
    model = load_model()
    con = sqlite3.connect(db_path)
    new = pd.read_sql('SELECT * FROM incidents WHERE opened_at >= ? AND opened_at < ?', con, params=[start, end])
    if new.empty:
        con.close()
        return new
    res = model.predict(new)
    res.insert(0, 'number', new['number'].values)
    res.insert(1, 'scored_at', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    res.insert(2, 'model_version', model.meta['model_version'])
    con.executemany('INSERT OR REPLACE INTO risk_scores VALUES (?,?,?,?,?,?,?)', res.itertuples(index=False, name=None))
    con.commit()
    con.close()
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2016-05-02')
    ap.add_argument('--end', default='2016-06-01')
    a = ap.parse_args()
    r = score_window(a.start, a.end)
    print(f'scored {len(r)} incidents'); print(r['risk_band'].value_counts().to_string() if len(r) else '')

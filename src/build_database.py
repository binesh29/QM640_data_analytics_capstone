"""Build the SQLite database data/db/itsm.sqlite from the CSV files.

Tables
  event_log      raw UCI event log, 141,712 rows x 36 columns ('?' stored as NULL)
  incidents      one row per incident (output of notebook 01), 24,918 rows
  risk_scores    written by src/score_batch.py (one row per scored incident)
Views
  v_open_queue   incidents joined to their latest risk score, highest risk first

Run from the repository root:  python src/build_database.py
"""
import sqlite3
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sla_model import ROOT, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_scores (
    number              TEXT NOT NULL,
    scored_at           TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    breach_probability  REAL NOT NULL,
    risk_score_raw      REAL NOT NULL,
    risk_band           TEXT NOT NULL CHECK (risk_band IN ('High', 'Medium', 'Low')),
    suggested_action    TEXT,
    PRIMARY KEY (number, model_version)
);
CREATE INDEX IF NOT EXISTS ix_incidents_opened ON incidents(opened_at);
CREATE INDEX IF NOT EXISTS ix_incidents_group ON incidents(log_assignment_group);
CREATE INDEX IF NOT EXISTS ix_event_number ON event_log(number);
DROP VIEW IF EXISTS v_open_queue;
CREATE VIEW v_open_queue AS
SELECT i.number, i.opened_at, i.log_priority AS priority, i.log_category AS category,
       i.log_assignment_group AS assignment_group, r.breach_probability, r.risk_band, r.suggested_action,
       r.model_version, i.sla_breach AS actual_breach
FROM incidents i JOIN risk_scores r ON r.number = i.number
ORDER BY r.risk_score_raw DESC;
"""


def build(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    raw = pd.read_csv(ROOT / 'data' / 'raw' / 'incident_event_log.csv', na_values='?', low_memory=False)
    raw.to_sql('event_log', con, index=False, chunksize=20000)
    inc = pd.read_csv(ROOT / 'data' / 'processed' / 'incidents.csv')
    inc.to_sql('incidents', con, index=False)
    con.executescript(SCHEMA)
    con.commit()
    counts = {t: con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('event_log', 'incidents', 'risk_scores')}
    con.close()
    return counts


if __name__ == '__main__':
    print(build())

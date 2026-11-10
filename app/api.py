"""REST scoring service for the SLA breach risk model (FastAPI).

ServiceNow (or any client) posts the at-logging fields of a new incident and receives the breach probability,
risk band and suggested action. Run from the repository root:
    uvicorn app.api:app --port 8000
Then open http://localhost:8000/docs for the interactive API page.
"""
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sla_model import load_model

model = load_model()
app = FastAPI(title='SLA breach risk service', version=model.meta['model_version'])


class Incident(BaseModel):
    number: str
    opened_at: str
    log_priority: str
    log_impact: str
    log_urgency: str
    log_category: Optional[str] = None
    log_subcategory: Optional[str] = None
    log_u_symptom: Optional[str] = None
    log_contact_type: Optional[str] = None
    log_location: Optional[str] = None
    log_assignment_group: Optional[str] = None
    log_notify: Optional[str] = None
    log_caller_id: Optional[str] = None
    log_cmdb_ci: Optional[str] = None
    log_u_priority_confirmation: bool = False


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': model.meta['model_version']}


@app.post('/score')
def score(incidents: list[Incident]):
    df = pd.DataFrame([i.model_dump() for i in incidents])
    out = model.predict(df)
    out.insert(0, 'number', df['number'].values)
    return out.to_dict('records')

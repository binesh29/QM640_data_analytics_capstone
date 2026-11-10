"""Service-desk risk dashboard (Streamlit).

Shows the queue of scored incidents for a selected day, highest risk first, with filters by assignment group and
risk band, plus a form to score a single new ticket. Reads from data/db/itsm.sqlite (risk_scores, v_open_queue).
Run from the repository root:
    streamlit run app/dashboard.py
"""
import sqlite3
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sla_model import DB_PATH, load_model

st.set_page_config(page_title='SLA breach risk', layout='wide')
model = load_model()
con = sqlite3.connect(DB_PATH)
days = pd.read_sql("SELECT DISTINCT substr(opened_at, 1, 10) AS d FROM v_open_queue ORDER BY d", con)['d'].tolist()

st.title('SLA breach risk: incident queue')
st.caption(f"Model {model.meta['model_version']} · random forest with recalibrated probabilities · "
           'scores use only fields known when the ticket is logged')
if not days:
    st.warning('No scored incidents yet. Run: python src/score_batch.py --start 2016-05-02 --end 2016-06-01')
    st.stop()

c1, c2, c3 = st.columns([1, 2, 1])
day = c1.selectbox('Day logged', days, index=min(len(days) - 1, days.index('2016-05-03') if '2016-05-03' in days else 0))
q = pd.read_sql('SELECT * FROM v_open_queue WHERE substr(opened_at, 1, 10) = ?', con, params=[day])
groups = c2.multiselect('Assignment group', sorted(q['assignment_group'].dropna().unique()))
bands = c3.multiselect('Risk band', ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
if groups:
    q = q[q['assignment_group'].isin(groups)]
q = q[q['risk_band'].isin(bands)]

k1, k2, k3, k4 = st.columns(4)
k1.metric('Tickets logged', f'{len(q):,}')
k2.metric('Flagged High', f"{(q['risk_band'] == 'High').sum():,}")
k3.metric('Expected breaches', f"{q['breach_probability'].sum():.0f}")
k4.metric('Mean breach probability', f"{q['breach_probability'].mean():.0%}" if len(q) else '–')

show = q[['number', 'opened_at', 'priority', 'assignment_group', 'category', 'breach_probability', 'risk_band', 'suggested_action']]
st.dataframe(show.rename(columns=lambda c: c.replace('_', ' ').capitalize()), hide_index=True, use_container_width=True,
             column_config={'Breach probability': st.column_config.ProgressColumn(format='%.2f', min_value=0, max_value=1)})

with st.expander('Score a new ticket'):
    with st.form('new'):
        a, b, c = st.columns(3)
        pr = a.selectbox('Priority', ['1 - Critical', '2 - High', '3 - Moderate', '4 - Low'], index=2)
        im = b.selectbox('Impact', ['1 - High', '2 - Medium', '3 - Low'], index=1)
        ur = c.selectbox('Urgency', ['1 - High', '2 - Medium', '3 - Low'], index=1)
        grp = a.text_input('Assignment group', 'Group 70')
        cat = b.text_input('Category', 'Category 26')
        ct = c.selectbox('Contact type', ['Phone', 'Email', 'Self service', 'Direct opening', 'IVR'])
        go = st.form_submit_button('Score')
    if go:
        t = pd.DataFrame([{'opened_at': pd.Timestamp.now(), 'log_priority': pr, 'log_impact': im, 'log_urgency': ur,
                           'log_assignment_group': grp, 'log_category': cat, 'log_contact_type': ct, 'log_subcategory': None,
                           'log_u_symptom': None, 'log_location': None, 'log_notify': 'Do Not Notify', 'log_caller_id': None,
                           'log_cmdb_ci': None, 'log_u_priority_confirmation': False}])
        r = model.predict(t).iloc[0]
        st.write(f"**Breach probability {r['breach_probability']:.0%} · {r['risk_band']} risk** – {r['suggested_action']}")
con.close()

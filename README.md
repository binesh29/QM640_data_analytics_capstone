# QM640 Data Analytics Capstone

**Predicting Service Level Agreement (SLA) Breach in IT Incident Management**

Dependent variable for all research questions: **SLA breach** (1 = breached, 0 = met).

Binesh Balakrishnan · Walsh College · QM640 Data Analytics Capstone · Term 3
Mentor: Ms Keya Choudhury Ganguli

## Research questions

| RQ | Question | Primary test / model |
|---|---|---|
| RQ1 | Does the SLA breach rate differ across the four priority levels? | χ² test of independence (4 × 2); pairwise χ² (Holm) |
| RQ2 | Is reassignment associated with SLA breach, controlling for priority and category? | χ² test; multivariable logistic regression |
| RQ3 | Is knowledge-base use associated with SLA breach, controlling for priority and category? | χ² test; multivariable logistic regression |
| RQ4 | Does an ML model using at-logging attributes beat a priority-only baseline on AUC? | Logistic regression, random forest, XGBoost; DeLong test |

## Data

- **Source:** Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H
- **Page:** https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0). The raw file is redistributed here unchanged, with attribution.
- **Real, anonymised ServiceNow audit data** – not synthetic, not from Kaggle.

| Level | Rows | Notes |
|---|---|---|
| Raw event log | 141,712 events × 36 columns | 24,918 incidents; missing values coded `?` |
| Incident level | 24,918 incidents | used for all four RQs (outcome: SLA breach) |
| Valid resolution time | 23,323 incidents | descriptive only (not an outcome) |

Key profile figures: SLA breach prevalence 36.6% (9,115 incidents); 94.2% of incidents are Moderate priority; 98.1% were opened March–May 2016. `made_sla = true` means the SLA was **met** (median resolution 1.1 h vs 201.7 h when false).

## Repository structure

```
QM640_data_analytics_capstone/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── incident_event_log.csv      # UCI source file, unchanged
│   └── processed/
│       └── incidents.csv               # one row per incident (created by notebook 01)
├── notebooks/
│   └── 01_data_preparation.ipynb       # load, validate, collapse, derive, profile
├── src/
│   └── sample_size.py                  # minimum sample size per RQ (synopsis Tables 3–4)
└── docs/
    ├── QM640_Synopsis_Balakrishnan_Draft.docx
    └── QM640_Synopsis_Balakrishnan_Draft.pdf
```

Notebooks for RQ1–RQ4 will be added after the synopsis is approved.

## Sample size (α = 0.05, power = 0.80)

| RQ | Method | Minimum N |
|---|---|---|
| RQ1 | χ² test of independence, w = 0.10, df = 3 | 1,091 |
| RQ2 | χ² test of independence, w = 0.10, df = 1, adjusted for covariates (÷ (1 − R²), R² = 0.047) | 824 |
| RQ3 | χ² test of independence, w = 0.10, df = 1, adjusted for covariates (÷ (1 − R²), R² = 0.154) | 928 |
| RQ4 | AUC precision (Hanley & McNeil, 1982), AUC 0.75, e = 0.03, p = 0.366, 30% hold-out | 3,970 |

Final minimum N = 3,970; available N = 24,918 (6.3×). Run `python src/sample_size.py` to reproduce.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook 01_data_preparation.ipynb
```

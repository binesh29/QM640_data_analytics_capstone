# QM640 Data Analytics Capstone

**Predicting Service Level Agreement (SLA) Breach and Resolution Time in IT Incident Management**

Binesh Balakrishnan · Walsh College · QM640 Data Analytics Capstone · Term 3
Mentor: Ms Keya Choudhury Ganguli

## Research questions

| RQ | Question | Primary test / model |
|---|---|---|
| RQ1 | Does resolution time differ across the four priority levels? | Kruskal–Wallis H, Dunn post hoc (Holm) |
| RQ2 | Is reassignment associated with SLA breach, controlling for priority and category? | χ² test; multivariable logistic regression |
| RQ3 | Is knowledge-base use associated with a difference in resolution time? | Mann–Whitney U; adjusted log-linear regression |
| RQ4 | Does an ML model using at-logging attributes beat a priority-only baseline on AUC? | Logistic regression, random forest, XGBoost; DeLong test |

## Data

- **Source:** Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H
- **Page:** https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0). The raw file is redistributed here unchanged, with attribution.
- **Real, anonymised ServiceNow audit data** – not synthetic, not from Kaggle.

| Level | Rows | Notes |
|---|---|---|
| Raw event log | 141,712 events × 36 columns | 24,918 incidents; missing values coded `?` |
| Incident level | 24,918 incidents | used for RQ2 and RQ4 |
| Valid resolution time | 23,323 incidents | excludes 1,556 with no `resolved_at` and 39 with zero duration; used for RQ1 and RQ3 |

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
│   └── sample_size.py                  # minimum sample size per RQ (synopsis Table 2)
└── docs/
    └── QM640_Synopsis_Balakrishnan_Draft.docx
```

Notebooks for RQ1–RQ4 will be added after the synopsis is approved.

## Sample size (α = 0.05, power = 0.80)

| RQ | Method | Minimum N |
|---|---|---|
| RQ1 | One-way ANOVA, f = 0.10, k = 4, adjusted for Kruskal–Wallis (ARE 0.864) | 1,268 |
| RQ2 | χ² test of independence, w = 0.10, df = 1 | 785 |
| RQ3 | Two-sample t, d = 0.20, allocation 1:6 (observed 1:5.8), ARE 0.864 | 1,864 |
| RQ4 | AUC precision (Hanley & McNeil, 1982), AUC 0.75, e = 0.03, p = 0.366, 30% hold-out | 3,970 |

Final minimum N = 3,970; available N = 24,918 (6.3×). Run `python src/sample_size.py` to reproduce.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook 01_data_preparation.ipynb
```

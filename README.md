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
│   ├── raw/incident_event_log.csv          # UCI source file, unchanged
│   └── processed/incidents.csv             # one row per incident (notebook 01)
├── notebooks/
│   ├── 01_data_preparation.ipynb           # load, validate, collapse, derive
│   ├── 02_cleaning_and_eda.ipynb           # cleaning log, EDA figures
│   ├── 03_rq1_rq3_hypothesis_tests.ipynb   # chi-square, logistic regression, Mantel-Haenszel
│   └── 04_rq4_prediction_models.ipynb      # baseline, LR, RF, XGBoost, DeLong test
├── src/
│   ├── sample_size.py                      # minimum sample size per RQ
│   └── plot_style.py                       # shared chart style
├── figures/                                # report figures (PNG)
├── results/                                # numeric outputs (JSON) used in the reports
└── docs/
    ├── QM640_Synopsis_Balakrishnan_Final.docx / .pdf
    └── QM640_Interim_Report_Balakrishnan_Draft.docx / .pdf
```

## Preliminary results (interim report)

| RQ | Result | Decision |
|---|---|---|
| RQ1 | χ²(3) = 1,293.1, p < .001, Cramér's V = 0.228 | Reject H0 |
| RQ2 | Adjusted OR = 1.95 per reassignment (95% CI 1.90–2.00) | Reject H0 |
| RQ3 | Raw OR = 6.24; reverses to 0.61 after controlling for opening week (March 2016 structural break) | Reject H0; direction tentative |
| RQ4 | Random forest AUC 0.785 vs priority-only baseline 0.552 (DeLong p < .001) | Reject H0 |

## Sample size (α = 0.05, power = 0.80)

| RQ | Method | Minimum N |
|---|---|---|
| RQ1 | χ² test of independence, w = 0.10, df = 3 | 1,091 |
| RQ2 | χ² test of independence, w = 0.10, df = 1, adjusted for covariates (÷ (1 − R²), R² = 0.047) | 824 |
| RQ3 | χ² test of independence, w = 0.10, df = 1, adjusted for covariates (÷ (1 − R²), R² = 0.154) | 928 |
| RQ4 | AUC precision (Hanley & McNeil, 1982), AUC 0.75, e = 0.03, p = 0.366, 30% hold-out | 3,970 |

Final minimum N = 3,970; available N = 24,918 (6.3×). Run `python src/sample_size.py` to reproduce.

## Reproducibility note

Results in `results/` and in the reports were produced with Python 3.11 and the exact library versions pinned in `requirements.txt`, with random seed 42. Notebooks 02 and 03 gave identical results in a Google Colab run with different library versions. In notebook 04, XGBoost can vary by about ±0.015 test AUC across library versions, because two tree depths (5 and 7) score almost the same in cross-validation and a version change can tip the choice; for example, a Google Colab run selected depth 5 and gave a test AUC of 0.751 rather than 0.738. All other models vary by 0.001 AUC or less, and no conclusion changes.

To run in Google Colab, open a notebook from this repository; its first code cell clones the repository and moves into `notebooks/`.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_preparation.ipynb 02_cleaning_and_eda.ipynb \
    03_rq1_rq3_hypothesis_tests.ipynb 04_rq4_prediction_models.ipynb
```

Run the notebooks in order. Notebook 04 takes about 8 minutes on a laptop (grid search with 5-fold CV). Random seed = 42.

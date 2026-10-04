# QM640 capstone - minimum sample sizes (alpha = 0.05, power = 0.80)
# Dependent variable for all RQs: SLA breach (binary)
import numpy as np
from scipy.stats import chi2, ncx2

def chi2_n(w, df, a=0.05, p=0.80):            # chi-square test of independence
    n, c = 10, chi2.ppf(1 - a, df)
    while ncx2.sf(c, df, n * w**2) < p:
        n += 1
    return n

def auc_test_n(auc, e, prev):                 # RQ4 (Hanley & McNeil, 1982)
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    n = 50
    while True:
        pos = round(n * prev); neg = n - pos
        var = (auc*(1-auc) + (pos-1)*(q1-auc**2) + (neg-1)*(q2-auc**2)) / (pos*neg)
        if 1.96 * np.sqrt(var) <= e:
            return n
        n += 1

rq1 = chi2_n(0.10, 3)                         # priority (4) x breach (2): 1091
rq2 = int(np.ceil(chi2_n(0.10, 1) / (1 - 0.047)))  # 785 / 0.953 -> 824
rq3 = int(np.ceil(chi2_n(0.10, 1) / (1 - 0.154)))  # 785 / 0.846 -> 928
test = auc_test_n(0.75, 0.03, 9115 / 24918)   # observed prevalence 0.366: 1191
rq4 = int(np.ceil(test / 0.30))               # 3970
print(rq1, rq2, rq3, test, rq4, "final N =", max(rq1, rq2, rq3, rq4))

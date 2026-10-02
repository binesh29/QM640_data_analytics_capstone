# QM640 capstone - minimum sample sizes (alpha = 0.05, power = 0.80)
import numpy as np
from scipy.stats import f, ncf, chi2, ncx2, t, nct

ARE = 0.864  # minimum efficiency of rank tests vs F/t tests

def anova_n(fe, k, a=0.05, p=0.80):          # RQ1
    n = 2 * k
    while ncf.sf(f.ppf(1 - a, k - 1, n - k), k - 1, n - k, fe**2 * n) < p:
        n += 1
    return n

def chi2_n(w, df, a=0.05, p=0.80):            # RQ2
    n, c = 10, chi2.ppf(1 - a, df)
    while ncx2.sf(c, df, n * w**2) < p:
        n += 1
    return n

def ttest_n(d, ratio, a=0.05, p=0.80):        # RQ3 (n2 = ratio * n1)
    n1 = 2
    while True:
        n2 = int(np.ceil(ratio * n1)); df = n1 + n2 - 2
        ncp = d / np.sqrt(1 / n1 + 1 / n2); c = t.ppf(1 - a / 2, df)
        if nct.sf(c, df, ncp) + nct.cdf(-c, df, ncp) >= p:
            return n1 + n2
        n1 += 1

def auc_test_n(auc, e, prev):                 # RQ4 (Hanley & McNeil, 1982)
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    n = 50
    while True:
        pos = round(n * prev); neg = n - pos
        var = (auc*(1-auc) + (pos-1)*(q1-auc**2) + (neg-1)*(q2-auc**2)) / (pos*neg)
        if 1.96 * np.sqrt(var) <= e:
            return n
        n += 1

rq1 = int(np.ceil(anova_n(0.10, 4) / ARE))    # 1095 / 0.864 -> 1268
rq2 = chi2_n(0.10, 1)                         # 785
rq3 = int(np.ceil(ttest_n(0.20, 6) / ARE))    # observed 1:5.8 -> 1:6; 1610 / 0.864 -> 1864
test = auc_test_n(0.75, 0.03, 9115 / 24918)   # observed prevalence 0.366; 1191
rq4 = int(np.ceil(test / 0.30))               # 3970
print(rq1, rq2, rq3, test, rq4, "final N =", max(rq1, rq2, rq3, rq4))

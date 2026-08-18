"""
Dev-only verification tool: cross-checks part_a.py's closed-form OLS and
part_b.py's closed-form ridge against sklearn's LinearRegression/Ridge on
the same data.

Why this is needed: eval_a.py/eval_b.py's "Predictions match submitted
weights" check only proves predictions.txt was honestly derived from
weights.txt -- it never verifies the weights themselves solve the right
equation. This script gives that missing check, on a fast in-memory
subset instead of the full ~1.18GB CSV.

sklearn's Ridge objective is ||y - Xw||^2 + alpha * ||w||^2 (no 1/2
factors, no averaging by n), and fit_intercept=True does not penalize the
intercept -- both match the assignment's (1/2)sum(resid^2) + (lambda/2)
sum(w_j^2) with the intercept excluded from R, once the 1/2 factors cancel
in the objective's overall scale. So alpha=lambda is the right mapping.

NOT part of the graded submission -- do not include in the Moodle zip.

usage: python3 verify_ab.py [n_rows]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

import part_a_dev
import part_b

TRAIN_CSV = "PartaData/e4_hr_train_downsampled.csv"
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


def load_subset(n_rows):
    print(f"[verify] loading first {n_rows} rows of {TRAIN_CSV}")
    return pd.read_csv(TRAIN_CSV, nrows=n_rows)


def compare(name, manual_w, sk_intercept, sk_coef, tol=1e-6):
    manual_w = np.asarray(manual_w).flatten()
    manual_intercept = manual_w[0]
    manual_coef = manual_w[1:]
    d_intercept = abs(manual_intercept - sk_intercept)
    d_coef = np.abs(manual_coef - sk_coef)
    max_d = max(d_intercept, d_coef.max())
    scale = abs(sk_intercept) + np.abs(sk_coef).max() + 1e-12
    rel = max_d / scale
    ok = rel < tol
    print(f"[verify] {name:28s} max_abs_diff={max_d:12.3e}  max_rel_diff={rel:12.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    df = load_subset(N_ROWS)

    # --- Part (a): OLS ---
    X_aug, y = part_a_dev.get_X_y(df, target_col="hr")
    w_manual = part_a_dev.train(X_aug, y)

    X_raw = df.drop(columns=["hr"]).to_numpy()
    y_flat = df["hr"].to_numpy()

    sk = LinearRegression(fit_intercept=True)
    sk.fit(X_raw, y_flat)
    all_ok = compare("part_a (OLS)", w_manual, sk.intercept_, sk.coef_)

    # --- Part (b): Ridge, a few lambda values (0.0 should reduce to OLS) ---
    for lam in [0.0, 1.0, 10.0]:
        w_manual = part_b.train(X_aug, y, lam)
        sk = Ridge(alpha=lam, fit_intercept=True, solver="cholesky")
        sk.fit(X_raw, y_flat)
        ok = compare(f"part_b (Ridge, lambda={lam})", w_manual, sk.intercept_, sk.coef_)
        all_ok = all_ok and ok

    print()
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

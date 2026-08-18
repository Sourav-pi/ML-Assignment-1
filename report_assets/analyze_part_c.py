import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, ".")
import submission.part_c as pc

RNG = np.random.default_rng(42)

print("[analyze_part_c] loading full training CSV...")
df = pd.read_csv("PartaData/e4_hr_train_downsampled.csv")
n = len(df)
idx = RNG.permutation(n)
n_val = int(0.2 * n)
val_idx, train_idx = idx[:n_val], idx[n_val:]

y = df["hr"].to_numpy(dtype=float)
X_raw = df.drop(columns=["hr"]).to_numpy(dtype=float)

X_train_raw, y_train = X_raw[train_idx], y[train_idx]
X_val_raw, y_val = X_raw[val_idx], y[val_idx]
print(f"[analyze_part_c] train n={len(y_train)}  val n={len(y_val)}")

print("[analyze_part_c] building features...")
X_train = pc.create_features(X_train_raw)
X_val = pc.create_features(X_val_raw)
print(f"[analyze_part_c] feature count: {X_train.shape[1]}")

# ---- feature names, matching create_features' exact append order ----
fs_bvp, min_hr, max_hr = 64, 45, 190
lag_min = int(np.floor(fs_bvp * 60 / max_hr))
lag_max = int(np.ceil(fs_bvp * 60 / min_hr))
L = 640
n_lags = min(lag_max, L) - lag_min
feature_names = [f"acf_lag{lag}" for lag in range(lag_min, lag_min + n_lags)]
feature_names += ["acf_peak_value", "acf_peak_lag_samples", "acf_implied_hr"]
feature_names += ["bvp_mean", "bvp_std", "bvp_mean_abs_diff", "bvp_mean_sq_diff",
                   "bvp_p25", "bvp_p75", "bvp_range", "bvp_zero_crossing_rate"]
feature_names += ["acc_mag_sq_mean", "acc_mag_sq_std", "acc_mag_sq_mean_abs_diff",
                   "acc_x_std", "acc_y_std", "acc_z_std", "acc_mag_sq_max",
                   "acc_xy_corr", "acc_yz_corr", "acc_xz_corr"]
feature_names += ["eda_mean", "eda_std", "eda_range", "eda_slope"]
assert len(feature_names) == X_train.shape[1], (len(feature_names), X_train.shape[1])

print("[analyze_part_c] selecting alpha via 5-fold CV...")
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
best_alpha, cv_results = pc.select_alpha_kfold(X_train, y_train, alphas)
print(f"[analyze_part_c] best_alpha={best_alpha}")
for a, e in cv_results:
    print(f"    alpha={a:>6}  cv_nmae={e:.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = Ridge(alpha=best_alpha).fit(X_train_scaled, y_train)
pred_val = model.predict(X_val_scaled)
pred_median = np.full_like(y_val, np.median(y_train))

nmae_model = pc.NMAE(pred_val, y_val)
nmse_model = pc.NMSE(pred_val, y_val)
nmae_median = pc.NMAE(pred_median, y_val)
nmse_median = pc.NMSE(pred_median, y_val)

print(f"\n=== Part (c) local validation (n_val={len(y_val)}) ===")
print(f"{'':10s} {'NMAE':>10s} {'NMSE':>10s}")
print(f"{'model':10s} {nmae_model:10.4f} {nmse_model:10.4f}")
print(f"{'median':10s} {nmae_median:10.4f} {nmse_median:10.4f}")

# Top-5 by |standardized coefficient| -- model.coef_ IS the standardized
# coefficient since Ridge was fit on scaler-transformed features directly.
order = np.argsort(-np.abs(model.coef_))[:5]
print("\n=== top 5 features by |standardized coef| ===")
for i in order:
    print(f"{feature_names[i]:28s} std_coef={model.coef_[i]:+10.4f}")

np.save("report_assets/part_c_std_coef.npy", model.coef_)
with open("report_assets/part_c_feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))

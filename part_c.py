import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Normalized Mean Absolute Error
def NMAE(y_pred, y_real):
    n = y_real.shape[0]
    y_mean = np.mean(y_real)
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += abs(y_real[i] - y_pred[i])
        denominator += abs(y_real[i] - y_mean)
    return numerator / denominator

# Normalized Mean Squared Error
def NMSE(y_pred, y_real):
    n = y_real.shape[0]
    y_mean = np.mean(y_real)
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (y_real[i] - y_pred[i]) ** 2
        denominator += (y_real[i] - y_mean) ** 2
    return numerator / denominator

def slope(sig):
    T = sig.shape[1]
    t = np.arange(T) - (T - 1) / 2.0
    return (sig @ t).reshape(-1, 1) / np.sum(t ** 2)

def create_features(X_raw, fs_bvp=64, min_hr=45, max_hr=190):
    # stitch the ten one-second blocks back into full-length signals
    bvp_cols, accx_cols, accy_cols, accz_cols, eda_cols = [], [], [], [], []
    for s in range(10):
        b = s * 164
        accx_cols.append(X_raw[:, b:b + 32])
        accy_cols.append(X_raw[:, b + 32:b + 64])
        accz_cols.append(X_raw[:, b + 64:b + 96])
        bvp_cols.append(X_raw[:, b + 96:b + 160])
        eda_cols.append(X_raw[:, b + 160:b + 164])

    bvp = np.hstack(bvp_cols)     # n x 640
    acc_x = np.hstack(accx_cols)  # n x 320
    acc_y = np.hstack(accy_cols)
    acc_z = np.hstack(accz_cols)
    eda = np.hstack(eda_cols)     # n x 40

    feats = []

    # autocorrelation of the BVP window over the plausible HR lag range -
    # the dominant lag is basically the heart rate
    x = bvp - bvp.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True) + 1e-8
    lag_min = int(np.floor(fs_bvp * 60 / max_hr))
    lag_max = int(np.ceil(fs_bvp * 60 / min_hr))
    L = x.shape[1]
    acf_cols = []
    for lag in range(lag_min, min(lag_max, L)):
        acf_cols.append(np.mean(x[:, :-lag] * x[:, lag:], axis=1, keepdims=True) / var)
    acf = np.hstack(acf_cols)
    feats.append(acf)

    # peak height, peak lag, and the HR implied by the peak lag
    lags = np.arange(lag_min, lag_min + acf.shape[1])
    peak_idx = np.argmax(acf, axis=1)
    peak_val = acf[np.arange(acf.shape[0]), peak_idx].reshape(-1, 1)
    peak_lag = lags[peak_idx].reshape(-1, 1).astype(float)
    feats.append(peak_val)
    feats.append(peak_lag)
    feats.append(fs_bvp * 60.0 / peak_lag)

    # BVP statistical / shape features
    feats.append(bvp.mean(axis=1, keepdims=True))
    feats.append(bvp.std(axis=1, keepdims=True))
    feats.append(np.abs(np.diff(bvp, axis=1)).mean(axis=1, keepdims=True))
    feats.append((np.diff(bvp, axis=1) ** 2).mean(axis=1, keepdims=True))
    feats.append(np.percentile(bvp, 25, axis=1, keepdims=True))
    feats.append(np.percentile(bvp, 75, axis=1, keepdims=True))
    feats.append(bvp.max(axis=1, keepdims=True) - bvp.min(axis=1, keepdims=True))
    zc = np.sum(np.diff(np.sign(x), axis=1) != 0, axis=1, keepdims=True) / L
    feats.append(zc)

    # accelerometer features (activity level)
    acc_mag_sq = acc_x**2 + acc_y**2 + acc_z**2
    feats.append(acc_mag_sq.mean(axis=1, keepdims=True))
    feats.append(acc_mag_sq.std(axis=1, keepdims=True))
    feats.append(np.abs(np.diff(acc_mag_sq, axis=1)).mean(axis=1, keepdims=True))
    feats.append(acc_x.std(axis=1, keepdims=True))
    feats.append(acc_y.std(axis=1, keepdims=True))
    feats.append(acc_z.std(axis=1, keepdims=True))
    feats.append(acc_mag_sq.max(axis=1, keepdims=True))
    feats.append(np.mean(acc_x * acc_y, axis=1, keepdims=True))
    feats.append(np.mean(acc_y * acc_z, axis=1, keepdims=True))
    feats.append(np.mean(acc_x * acc_z, axis=1, keepdims=True))

    # EDA features (arousal)
    feats.append(eda.mean(axis=1, keepdims=True))
    feats.append(eda.std(axis=1, keepdims=True))
    feats.append(eda.max(axis=1, keepdims=True) - eda.min(axis=1, keepdims=True))
    feats.append(slope(eda))

    return np.hstack(feats)

def select_alpha_kfold(X, y, alphas, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_results = []
    best_alpha = None
    min_nmae = float('inf')

    for a in alphas:
        fold_errors = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

            model = Ridge(alpha=a).fit(X_tr, y_tr)
            fold_errors.append(NMAE(model.predict(X_va), y_va))

        nmae = np.mean(fold_errors)
        cv_results.append((a, nmae))

        if nmae < min_nmae:
            min_nmae = nmae
            best_alpha = a

    return best_alpha, cv_results

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_c.py train.csv test.csv predictions.txt")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    pred_path = sys.argv[3]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df.iloc[:, -1].values
    X_train_raw = train_df.iloc[:, :-1].values

    # test.csv might not include the hr column, so figure it out from the column count
    n_feat_cols = X_train_raw.shape[1]
    if test_df.shape[1] == n_feat_cols + 1:
        X_test_raw = test_df.iloc[:, :-1].values
    else:
        X_test_raw = test_df.values

    X_train = create_features(X_train_raw)
    X_test = create_features(X_test_raw)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    best_alpha, _ = select_alpha_kfold(X_train, y_train, alphas)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=best_alpha).fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    np.savetxt(pred_path, predictions, fmt='%f')

if __name__ == "__main__":
    main()

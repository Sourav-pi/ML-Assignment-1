import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW_IDX = 12000  # arbitrary, well into the file, avoids header/edge effects

df = pd.read_csv("PartaData/e4_hr_train_downsampled.csv", skiprows=range(1, ROW_IDX + 1), nrows=1)
row = df.iloc[0]
hr_target = row["hr"]

bvp_cols, accx_cols, accy_cols, accz_cols, eda_cols = [], [], [], [], []
for s in range(10):
    b = s * 164
    accx_cols.append(row.iloc[b:b + 32].to_numpy(dtype=float))
    accy_cols.append(row.iloc[b + 32:b + 64].to_numpy(dtype=float))
    accz_cols.append(row.iloc[b + 64:b + 96].to_numpy(dtype=float))
    bvp_cols.append(row.iloc[b + 96:b + 160].to_numpy(dtype=float))
    eda_cols.append(row.iloc[b + 160:b + 164].to_numpy(dtype=float))

bvp = np.concatenate(bvp_cols)
acc_x = np.concatenate(accx_cols)
acc_y = np.concatenate(accy_cols)
acc_z = np.concatenate(accz_cols)
eda = np.concatenate(eda_cols)

a_sq = acc_x ** 2 + acc_y ** 2 + acc_z ** 2
a_sq_norm = a_sq / np.mean(a_sq)

t_bvp = np.arange(bvp.size) / 64.0
t_acc = np.arange(a_sq.size) / 32.0
t_eda = np.arange(eda.size) / 4.0

fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.0), sharex=True)

axes[0].plot(t_bvp, bvp, color="#1f77b4", linewidth=0.9)
axes[0].set_ylabel("BVP\n(device units)")

axes[1].plot(t_acc, a_sq_norm, color="#ff7f0e", linewidth=0.9)
axes[1].set_ylabel(r"$\tilde{a}_{sq}(t)$" + "\n(unitless)")

axes[2].plot(t_eda, eda, color="#2ca02c", marker="o", markersize=3, linewidth=0.9)
axes[2].set_ylabel("EDA\n(microsiemens)")
axes[2].set_xlabel("Time (s)")

fig.suptitle(f"Part (a) representative window -- target heart rate = {hr_target:.1f} bpm")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("report_assets/part_a_window.pdf")
fig.savefig("report_assets/part_a_window.png", dpi=200)
print(f"row {ROW_IDX}: hr={hr_target:.2f}, bvp range=[{bvp.min():.3f},{bvp.max():.3f}], "
      f"a_sq_norm range=[{a_sq_norm.min():.3f},{a_sq_norm.max():.3f}], "
      f"eda range=[{eda.min():.3f},{eda.max():.3f}]")

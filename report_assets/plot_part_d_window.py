import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW_IDX = 300  # arbitrary, well into the file

with np.load("PartdData/random_train/train_1.npz", allow_pickle=False) as data:
    bvp = data["e4_bvp"][WINDOW_IDX].astype(float)
    acc = data["e4_acc"][WINDOW_IDX].astype(float)  # (9600, 3)
    eda = data["e4_eda"][WINDOW_IDX].astype(float)
    ecg = data["zephyr_ecg"][WINDOW_IDX].astype(float)
    hr = data["e4_hr"][WINDOW_IDX].astype(float)
    glucose = data["glucose"][WINDOW_IDX]

a_sq = np.nansum(acc ** 2, axis=1)
a_sq_norm = a_sq / np.nanmean(a_sq)

t_bvp = np.arange(bvp.size) / 64.0
t_acc = np.arange(a_sq.size) / 32.0
t_eda = np.arange(eda.size) / 4.0
t_ecg = np.arange(ecg.size) / 250.0
t_hr = np.arange(hr.size) / 1.0

fig, axes = plt.subplots(5, 1, figsize=(7.5, 10.5), sharex=True)

axes[0].plot(t_bvp, bvp, color="#1f77b4", linewidth=0.5)
axes[0].set_ylabel("BVP\n(device units)")

axes[1].plot(t_acc, a_sq_norm, color="#ff7f0e", linewidth=0.5)
axes[1].set_ylabel(r"$\tilde{a}_{sq}(t)$" + "\n(unitless)")

axes[2].plot(t_eda, eda, color="#2ca02c", linewidth=0.8)
axes[2].set_ylabel("EDA\n(microsiemens)")

axes[3].plot(t_ecg, ecg, color="#d62728", linewidth=0.3)
axes[3].set_ylabel("ECG\n(device units)")

axes[4].plot(t_hr, hr, color="#9467bd", linewidth=1.0)
axes[4].set_ylabel("Heart rate\n(bpm)")
axes[4].set_xlabel("Time (s)")

fig.suptitle(f"Part (d) representative window -- target CGM glucose = {glucose:.0f} mg/dL")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("report_assets/part_d_window.pdf")
fig.savefig("report_assets/part_d_window.png", dpi=200)
print(f"glucose={glucose:.1f}, bvp range=[{np.nanmin(bvp):.2f},{np.nanmax(bvp):.2f}], "
      f"a_sq_norm range=[{np.nanmin(a_sq_norm):.3f},{np.nanmax(a_sq_norm):.3f}], "
      f"eda range=[{np.nanmin(eda):.3f},{np.nanmax(eda):.3f}], "
      f"ecg range=[{np.nanmin(ecg):.1f},{np.nanmax(ecg):.1f}], "
      f"hr range=[{np.nanmin(hr):.1f},{np.nanmax(hr):.1f}]")

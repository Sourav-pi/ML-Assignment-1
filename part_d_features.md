# Part (d) feature dictionary

All 103 features currently computed by `make_features_for_window` in `part_d.py`, in
the exact order they appear in `feature_names`/`coef`. No pairwise/interaction or
quadratic terms are included (both were tried and removed — see git history and
conversation notes: quadratic terms were never selected by Lasso in any protocol,
and pairwise interactions net-hurt d3 while only marginally helping d1).

Every feature is NaN-safe: on a degenerate window (too much missing data, too few
detected peaks/beats, a numerically degenerate computation) the underlying function
emits `np.nan` rather than a fabricated fallback value. NaNs are imputed with the
per-feature training-set median (`SimpleImputer`, fit on train only) in `cmd_train`.

Two calibration-offset fixes apply throughout: raw `zephyr_ecg`, `zephyr_breathing`,
and `e4_temp` are unscaled device counts with large, subject-specific DC offsets
that otherwise act as a subject fingerprint (a leakage risk, confirmed empirically —
these channels' raw mean/min/max topped the d3 cross-subject correlation table
before the fix). `_detrend`/`_stats_detrended` remove a slow moving-average
baseline before computing mean/std/min/max/range on these three channels; `_slope`
and `_block_trend_features` are untouched since both are already invariant to a
constant additive offset.

---

## BVP — E4 photoplethysmography (features 1–20)

Interpolated once (`_interpolate_nans`) and peak-detected once (`_bvp_peaks`) per
window; both are reused across every BVP-derived feature below.

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 1 | `bvp_mean` | BVP Mean Amplitude | Baseline waveform level (5-number summary, `_stats`). |
| 2 | `bvp_std` | BVP Amplitude Standard Deviation | Overall waveform variability. |
| 3 | `bvp_min` | BVP Minimum Amplitude | Trough level. |
| 4 | `bvp_max` | BVP Maximum Amplitude | Peak level. |
| 5 | `bvp_range` | BVP Amplitude Range | Peak-to-trough swing. |
| 6 | `bvp_slope` | BVP Linear Trend Slope | Linear drift of the raw waveform across the window. |
| 7 | `bvp_prv_sdnn` | Pulse-Rate Variability (SDNN) | PPG analogue of HRV: overall beat-interval variability. |
| 8 | `bvp_prv_rmssd` | Pulse-Rate Variability (RMSSD) | PRV beat-to-beat variability. |
| 9 | `bvp_rise_time_mean` | Mean Pulse Rise Time | Mean systolic upstroke duration (trough→peak); vascular tone/blood-viscosity shifts change how sharply pulses rise. |
| 10 | `bvp_pulse_amp_mean` | Mean Pulse Amplitude | Mean pulse amplitude (trough→peak height). |
| 11 | `bvp_wavelet_energy_l1` | BVP Wavelet Energy Fraction — Level 1 (finest) | Multi-scale energy fraction from a 6-level `db4` wavelet decomposition (PyWavelets) — a non-Fourier lens on "sharp vs. smooth" waveform structure across time-scales. |
| 12 | `bvp_wavelet_energy_l2` | BVP Wavelet Energy Fraction — Level 2 | Same technique, next-coarser scale. |
| 13 | `bvp_wavelet_energy_l3` | BVP Wavelet Energy Fraction — Level 3 | Same technique, next-coarser scale. |
| 14 | `bvp_wavelet_energy_l4` | BVP Wavelet Energy Fraction — Level 4 | Same technique, next-coarser scale. |
| 15 | `bvp_wavelet_energy_l5` | BVP Wavelet Energy Fraction — Level 5 | Same technique, next-coarser scale. |
| 16 | `bvp_wavelet_energy_l6` | BVP Wavelet Energy Fraction — Level 6 (coarsest) | Slowest, most macroscopic waveform component; the level most consistently selected by Lasso across protocols. |
| 17 | `bvp_augmentation_index` | PPG Augmentation Index | Ratio of the reflected-wave (late systolic) peak to the main pulse peak — an established arterial-stiffness/vascular-tone marker. |
| 18 | `bvp_teager_energy_mean` | Mean Teager–Kaiser Energy (BVP) | `x[n]² − x[n−1]·x[n+1]`, jointly sensitive to instantaneous amplitude and frequency — cited directly in Monte-Moreno (2011), the PPG-glucose reference in the assignment PDF. |
| 19 | `bvp_teager_energy_std` | Teager–Kaiser Energy Std. Dev. (BVP) | Variability of the same operator across the window. |
| 20 | `bvp_spectral_entropy` | BVP Spectral Entropy | Shannon entropy of the normalized Welch PSD — another Monte-Moreno (2011) feature category; low = periodic/regular, high = noisy/irregular. |

## E4 device-derived heart rate (features 21–27)

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 21 | `e4_hr_mean` | Mean Device Heart Rate | 5-number summary of the E4's own on-device HR estimate. |
| 22 | `e4_hr_std` | Device Heart Rate Std. Dev. | Overall on-device HR variability. |
| 23 | `e4_hr_min` | Minimum Device Heart Rate | Trough on-device HR. |
| 24 | `e4_hr_max` | Maximum Device Heart Rate | Peak on-device HR. |
| 25 | `e4_hr_range` | Device Heart Rate Range | Peak-to-trough on-device HR swing. |
| 26 | `e4_hr_block_trend` | Device Heart Rate Block Trend | 10-block within-window trend — captures *when* HR shifts, not just net drift (`_block_trend_features`). |
| 27 | `e4_hr_block_var` | Device Heart Rate Block Variability | Block-to-block variability. |

## EDA — electrodermal activity (features 28–48)

Sweating (adrenergic sweat-gland activation) is a textbook acute-hypoglycemia
symptom, motivating SCR (skin-conductance-response) *event* features specifically,
not just the mean EDA level.

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 28 | `eda_mean` | Mean Electrodermal Activity | Baseline EDA level (5-number summary). |
| 29 | `eda_std` | EDA Standard Deviation | Overall EDA variability. |
| 30 | `eda_min` | Minimum EDA | Trough EDA level. |
| 31 | `eda_max` | Maximum EDA | Peak EDA level. |
| 32 | `eda_range` | EDA Range | Peak-to-trough EDA swing. |
| 33 | `eda_slope` | EDA Linear Trend Slope | Linear drift in EDA level across the window. |
| 34 | `eda_scr_count` | Skin-Conductance-Response Count | Number of detected phasic SCR spikes. |
| 35 | `eda_scr_amp_mean` | Mean SCR Amplitude | Average size of detected SCR spikes. |
| 36 | `eda_scr_amp_sum` | Total SCR Amplitude | Summed size of all SCR spikes in the window. |
| 37 | `eda_tonic_slope` | EDA Tonic-Component Slope | Drift of the slow tonic (baseline arousal) component, isolated via moving-average subtraction. |
| 38 | `eda_time_since_last_scr` | Time Since Last SCR Event | Recency of the most recent SCR relative to window end — "how long ago was the last arousal event." Windows with zero SCRs get the full window duration (informative, not NaN). |
| 39 | `eda_scr_habituation` | SCR Habituation Slope | Slope of successive SCR amplitudes in temporal order — negative = diminishing sympathetic response (adaptation/habituation). |
| 40 | `eda_sampen` | EDA Sample Entropy | Complexity/regularity marker distinct from any single summary stat. |
| 41 | `eda_block_trend` | EDA Block Trend | Within-window block-based trend. |
| 42 | `eda_block_var` | EDA Block Variability | Block-to-block variability. |
| 43 | `eda_wavelet_energy_l1` | EDA Wavelet Energy Fraction — Level 1 (finest) | Multi-scale wavelet energy fraction, same technique as BVP. |
| 44 | `eda_wavelet_energy_l2` | EDA Wavelet Energy Fraction — Level 2 | Same technique, next-coarser scale. |
| 45 | `eda_wavelet_energy_l3` | EDA Wavelet Energy Fraction — Level 3 | Same technique, next-coarser scale. |
| 46 | `eda_wavelet_energy_l4` | EDA Wavelet Energy Fraction — Level 4 | Same technique, next-coarser scale. |
| 47 | `eda_wavelet_energy_l5` | EDA Wavelet Energy Fraction — Level 5 | Same technique, next-coarser scale. |
| 48 | `eda_wavelet_energy_l6` | EDA Wavelet Energy Fraction — Level 6 (coarsest) | Same technique, coarsest scale. |

## Temperature — E4 skin temperature (features 49–56)

Uses **detrended** stats (see calibration-offset note above) — raw skin temperature
is a stable per-person trait that otherwise leaks subject identity.

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 49 | `temp_mean` | Mean Skin Temperature (Detrended) | Within-window shape, not absolute per-subject baseline. |
| 50 | `temp_std` | Skin Temperature Std. Dev. (Detrended) | Within-window temperature variability. |
| 51 | `temp_min` | Minimum Skin Temperature (Detrended) | Trough within-window temperature. |
| 52 | `temp_max` | Maximum Skin Temperature (Detrended) | Peak within-window temperature. |
| 53 | `temp_range` | Skin Temperature Range (Detrended) | Peak-to-trough temperature swing. |
| 54 | `temp_slope` | Skin Temperature Linear Trend Slope | Linear drift. |
| 55 | `temp_block_trend` | Skin Temperature Block Trend | Within-window block-based trend. |
| 56 | `temp_block_var` | Skin Temperature Block Variability | Block-to-block variability. |

## E4 accelerometer (features 57–62)

Squared magnitude `a_sq(t) = ax(t)² + ay(t)² + az(t)²`.

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 57 | `e4_acc_sq_mean` | Mean Wrist Motion Energy | Baseline wrist-motion level (5-number summary). |
| 58 | `e4_acc_sq_std` | Wrist Motion Energy Std. Dev. | Overall wrist-motion variability. |
| 59 | `e4_acc_sq_min` | Minimum Wrist Motion Energy | Trough wrist-motion level. |
| 60 | `e4_acc_sq_max` | Maximum Wrist Motion Energy | Peak wrist-motion level. |
| 61 | `e4_acc_sq_range` | Wrist Motion Energy Range | Peak-to-trough wrist-motion swing. |
| 62 | `bvp_motion_quality` | BVP Signal Quality (Motion-Adjusted) | `bvp_std / (e4_acc_sq_mean + ε)` — how much to trust the BVP features in this window given concurrent wrist-motion artifact. |

## Zephyr ECG (features 63–83)

Interpolated once; R-peaks detected once via NeuroKit2 (`_ecg_r_peaks_nk`) and
reused by every peak-based feature below. Uses **detrended** stats (mean/std/min/max
carry a large per-subject DC offset otherwise — see calibration-offset note).

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 63 | `ecg_mean` | Mean ECG Amplitude (Detrended) | Within-window ECG shape, not per-subject calibration offset. |
| 64 | `ecg_std` | ECG Amplitude Std. Dev. (Detrended) | Within-window ECG amplitude variability. |
| 65 | `ecg_min` | Minimum ECG Amplitude (Detrended) | Trough within-window ECG amplitude — the single most consistently selected feature besides `ecg_qt_mean`. |
| 66 | `ecg_max` | Maximum ECG Amplitude (Detrended) | Peak within-window ECG amplitude. |
| 67 | `ecg_range` | ECG Amplitude Range (Detrended) | Peak-to-trough ECG amplitude swing. |
| 68 | `ecg_hr_mean` | Mean R-Peak-Derived Heart Rate | Heart rate computed directly from R-peak timing. |
| 69 | `ecg_sdnn` | Heart-Rate Variability (SDNN) | Overall RR-interval variability. |
| 70 | `ecg_rmssd` | Heart-Rate Variability (RMSSD) | Beat-to-beat RR-interval variability, sensitive to sympathetic/parasympathetic shifts. |
| 71 | `ecg_lf_power` | Low-Frequency HRV Power | Lomb-Scargle spectral power, 0.04–0.15Hz band (baroreflex/sympathetic-linked). |
| 72 | `ecg_hf_power` | High-Frequency HRV Power | Lomb-Scargle spectral power, 0.15–0.4Hz band (respiratory/parasympathetic-linked). |
| 73 | `ecg_lf_hf_ratio` | LF/HF Ratio (Sympathovagal Balance) | Standard clinical marker of autonomic balance; one of the most consistently Lasso-selected features. |
| 74 | `ecg_sd1` | Poincaré SD1 (Short-Term HRV) | Beat-to-beat variability, ⊥ line of identity — closely related to RMSSD/parasympathetic tone. |
| 75 | `ecg_sd2` | Poincaré SD2 (Long-Term HRV) | Longer-term variability, ∥ line of identity — closer to SDNN-style variability. |
| 76 | `ecg_sd1_sd2_ratio` | Poincaré SD1/SD2 Ratio | Geometric/nonlinear HRV summary distinct from the linear time/frequency metrics above; consistently Lasso-selected. |
| 77 | `ecg_qt_mean` | Mean QT-Interval Proxy | R-to-T-peak interval — cheap proxy for clinical QT (full delineation ~170× too slow at this scale). Motivated by hypoglycemia-induced QT prolongation, documented in Type 1 diabetics ("dead-in-bed syndrome" literature). The single most consistently Lasso-selected feature across every protocol. |
| 78 | `ecg_qtc_mean` | Mean Corrected QT Interval (Fridericia) | `QT / RR^(1/3)` — more accurate than Bazett's for hypoglycemia-related QT changes, generally more heart-rate-independent. |
| 79 | `ecg_qt_std` | QT-Interval Std. Dev. (Beat-to-Beat) | Beat-to-beat QT variability. |
| 80 | `ecg_qt_range` | QT-Interval Dispersion (Range) | Documented to increase during hypoglycemia. |
| 81 | `ecg_t_amplitude_mean` | Mean T-Wave Amplitude | Measured vs. each beat's local isoelectric baseline. T-wave flattening during hypoglycemia is catecholamine-mediated and documented in Type 1 diabetics. |
| 82 | `ecg_t_r_ratio` | T-Wave/R-Wave Amplitude Ratio | Additionally cancels a per-subject amplifier gain factor a plain amplitude wouldn't. |
| 83 | `ecg_sampen` | ECG RR-Interval Sample Entropy | Reduced HRV entropy is a documented marker of diabetic cardiac autonomic neuropathy. |

## Pulse transit time (features 84–85)

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 84 | `ptt_mean` | Mean Pulse Transit Time | Delay between each ECG R-peak and the BVP peak it produces peripherally — a validated arterial-stiffness/vascular-tone proxy, more direct than either signal's HRV/PRV alone since it measures actual pulse-wave travel time. |
| 85 | `ptt_std` | Pulse Transit Time Std. Dev. | Beat-to-beat variability of the same delay. |

## Zephyr accelerometer (features 86–91)

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 86 | `zephyr_acc_sq_mean` | Mean Chest Motion Energy | Baseline chest-strap motion level (5-number summary). |
| 87 | `zephyr_acc_sq_std` | Chest Motion Energy Std. Dev. | Overall chest-motion variability. |
| 88 | `zephyr_acc_sq_min` | Minimum Chest Motion Energy | Trough chest-motion level. |
| 89 | `zephyr_acc_sq_max` | Maximum Chest Motion Energy | Peak chest-motion level. |
| 90 | `zephyr_acc_sq_range` | Chest Motion Energy Range | Peak-to-trough chest-motion swing. |
| 91 | `ecg_motion_quality` | ECG Signal Quality (Motion-Adjusted) | `ecg_std / (zephyr_acc_sq_mean + ε)` — same signal-quality-vs-motion rationale as `bvp_motion_quality`, for the chest ECG. |

## Zephyr breathing (features 92–103)

Uses **detrended** stats (same calibration-offset rationale as ECG/temperature).
`block_trend`/`block_var` use the *raw* signal — both are already offset-invariant.

| # | Feature | Full name | Reasoning |
|---|---|---|---|
| 92 | `breathing_mean` | Mean Breathing Signal Amplitude (Detrended) | Within-window breathing shape, not per-subject calibration offset. |
| 93 | `breathing_std` | Breathing Signal Amplitude Std. Dev. (Detrended) | Within-window breathing-amplitude variability; consistently Lasso-selected. |
| 94 | `breathing_min` | Minimum Breathing Signal Amplitude (Detrended) | Trough within-window breathing amplitude. |
| 95 | `breathing_max` | Maximum Breathing Signal Amplitude (Detrended) | Peak within-window breathing amplitude. |
| 96 | `breathing_range` | Breathing Signal Amplitude Range (Detrended) | Peak-to-trough breathing-amplitude swing. |
| 97 | `breathing_resp_rate` | Mean Respiratory Rate | Instantaneous breathing rate from peak-to-peak timing, averaged across the window. |
| 98 | `breathing_resp_rate_std` | Respiratory Rate Std. Dev. | Variability of instantaneous breathing rate across the window. |
| 99 | `breathing_amp_mean` | Mean Breath Amplitude | Peak height above nearest preceding trough, averaged per breath. |
| 100 | `breathing_amp_std` | Breath Amplitude Std. Dev. | Variability of per-breath amplitude. |
| 101 | `breathing_block_trend` | Breathing Signal Block Trend | Within-window block-based trend. |
| 102 | `breathing_block_var` | Breathing Signal Block Variability | Block-to-block variability. |
| 103 | `ecg_rsa_power` | Respiratory Sinus Arrhythmia (RSA) Power | RR-interval spectral power centered on *this window's own measured breathing rate* (narrower/more targeted than the fixed HF band in `ecg_hf_power`). RSA magnitude indexes vagal tone; diabetic autonomic neuropathy is associated with reduced RSA. |

---

## What actually survives Lasso selection

The feature set above is intentionally broad (multi-modal exploration). What each
protocol's `LassoCV` fit actually keeps with nonzero weight is much sparser and is
protocol-dependent — see `diagnose compare`/`diagnose eval` output for current
numbers. Across every fit run so far, `ecg_qt_mean` (Mean QT-Interval Proxy) and
`ecg_min` (Minimum ECG Amplitude, Detrended) are the most consistently selected
features, with `ecg_lf_hf_ratio` (LF/HF Ratio), `ecg_sd1_sd2_ratio` (Poincaré
SD1/SD2 Ratio), and `breathing_std` (Breathing Signal Amplitude Std. Dev.)
appearing regularly. d1 (random within-subject) consistently selects far more
features (~70/103) than d2/d3 (~4/103 each) — expected, since d1's leakage-prone
split lets many weakly-correlated features look useful that a genuinely held-out
subject/time split screens out.

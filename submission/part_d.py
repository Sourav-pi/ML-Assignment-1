import numpy as np
import pickle
import sys
import warnings
from pathlib import Path

from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scipy.signal import find_peaks, lombscargle
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist

import neurokit2 as nk
import pywt

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

FORMAT_VERSION = 1


def _stats(x, prefix, names):
    """5-number summary (mean/std/min/max/range), NaN-safe."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        vals = [np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        vals = [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.max(x) - np.min(x)),
        ]
    names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max", f"{prefix}_range"])
    return vals


def peak_intervals(x, fs, min_rate_per_min, max_rate_per_min, prominence=None):
    """Peak indices and inter-peak intervals (seconds); prominence filters out noise-level peaks."""
    min_distance = int(fs * 60 / max_rate_per_min)
    peaks, _ = find_peaks(x, distance=min_distance, prominence=prominence)
    intervals = np.diff(peaks) / fs
    return peaks, intervals


def _slope(x, prefix, names):
    """Least-squares slope of a 1D array against sample index, ignoring NaNs."""
    idx = np.arange(x.shape[0])
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        val = np.nan
    else:
        val = float(np.polyfit(idx[mask], x[mask], 1)[0])
    names.append(f"{prefix}_slope")
    return [val]


def _interpolate_nans(x):
    """Linear interpolation over isolated missing samples; None if too much is missing."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    if valid.sum() < 0.5 * valid.size:
        return None
    idx = np.arange(x.size)
    return np.interp(idx, idx[valid], x[valid])


def _breathing_rate_features(x_filled, fs, names, min_rate_per_min=6, max_rate_per_min=40):
    """Instantaneous breathing rate and breath amplitude (mean/std) from peak-to-peak timing."""
    prefix = "breathing"
    feature_names = [f"{prefix}_resp_rate", f"{prefix}_resp_rate_std",
                      f"{prefix}_amp_mean", f"{prefix}_amp_std"]

    if x_filled is None:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan, np.nan]
    prominence = 0.5 * np.std(x_filled)

    peaks, intervals = peak_intervals(x_filled, fs=fs,
                                       min_rate_per_min=min_rate_per_min,
                                       max_rate_per_min=max_rate_per_min,
                                       prominence=prominence)

    if intervals.size < 2:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan, np.nan]

    inst_rate = 60.0 / intervals
    resp_rate = float(np.mean(inst_rate))
    resp_rate_std = float(np.std(inst_rate))

    # Amplitude: peak height above the nearest preceding trough, per breath.
    min_distance = max(1, int(fs * 60 / max_rate_per_min))
    troughs, _ = find_peaks(-x_filled, distance=min_distance, prominence=prominence)
    amps = []
    for p in peaks:
        prior_troughs = troughs[troughs < p]
        if prior_troughs.size:
            amps.append(x_filled[p] - x_filled[prior_troughs[-1]])

    amp_mean = float(np.mean(amps)) if amps else np.nan
    amp_std = float(np.std(amps)) if amps else np.nan

    names.extend(feature_names)
    return [resp_rate, resp_rate_std, amp_mean, amp_std]


def _moving_average(x, window):
    """O(N) moving average via uniform_filter1d; mode="nearest" avoids zero-padding edge bias."""
    if window <= 1:
        return x.copy()
    return uniform_filter1d(x, size=window, mode="nearest")


def _detrend(x_filled, fs, baseline_window_s=30):
    """Subtracts a slow moving-average baseline to remove each channel's per-subject calibration offset."""
    if x_filled is None:
        return None
    window = max(1, int(baseline_window_s * fs))
    if window >= x_filled.size:
        return x_filled - np.mean(x_filled)
    baseline = _moving_average(x_filled, window)
    return x_filled - baseline


def _stats_detrended(x_filled, fs, prefix, names, baseline_window_s=30):
    """Like _stats, but on the detrended signal (see _detrend)."""
    x_detrended = _detrend(x_filled, fs, baseline_window_s=baseline_window_s)
    if x_detrended is None:
        vals = [np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        vals = [
            float(np.mean(x_detrended)),
            float(np.std(x_detrended)),
            float(np.min(x_detrended)),
            float(np.max(x_detrended)),
            float(np.max(x_detrended) - np.min(x_detrended)),
        ]
    names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max", f"{prefix}_range"])
    return vals


def _block_trend_features(x, prefix, names, n_blocks=10):
    """Second-half-vs-first-half drift (block_trend) and block-to-block variability (block_std)."""
    feature_names = [f"{prefix}_block_trend", f"{prefix}_block_std"]
    x_filled = _interpolate_nans(x)
    if x_filled is None:
        names.extend(feature_names)
        return [np.nan, np.nan]

    blocks = np.array_split(x_filled, n_blocks)
    block_means = np.array([np.mean(b) for b in blocks])

    half = n_blocks // 2
    block_trend = float(np.mean(block_means[half:]) - np.mean(block_means[:half]))
    block_std = float(np.std(block_means))

    names.extend(feature_names)
    return [block_trend, block_std]


def _ecg_r_peaks_nk(x_filled, fs):
    """NeuroKit2 R-peak detection, run once and reused downstream; None on failure."""
    if x_filled is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _, info = nk.ecg_peaks(x_filled, sampling_rate=fs)
        r_peaks = np.asarray(info["ECG_R_Peaks"])
    except Exception:
        return None
    if r_peaks.size < 2:
        return None
    return r_peaks


def _hrv_features(r_peaks, fs, names):
    """R-peak-derived heart rate, SDNN, and RMSSD, computed directly with numpy (faster than nk.hrv_time)."""
    prefix = "ecg"
    feature_names = [f"{prefix}_hr_mean", f"{prefix}_sdnn", f"{prefix}_rmssd"]

    if r_peaks is None:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan]

    rr = np.diff(r_peaks) / fs
    hr_mean = float(np.mean(60.0 / rr))
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))

    names.extend(feature_names)
    return [hr_mean, sdnn, rmssd]


def _hrv_freq_features(r_peaks, fs, names, min_beats=20):
    """Frequency-domain HRV: LF (0.04-0.15Hz) / HF (0.15-0.4Hz) band power and their ratio, via Lomb-Scargle."""
    prefix = "ecg"
    feature_names = [f"{prefix}_lf_power", f"{prefix}_hf_power", f"{prefix}_lf_hf_ratio"]

    if r_peaks is None or r_peaks.size < min_beats:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan]

    beat_times = r_peaks[1:] / fs
    rr = np.diff(r_peaks) / fs
    rr_detrended = rr - np.mean(rr)

    lf_band = np.linspace(0.04, 0.15, 20) * 2 * np.pi
    hf_band = np.linspace(0.15, 0.4, 20) * 2 * np.pi

    try:
        lf_power = float(np.trapezoid(
            lombscargle(beat_times, rr_detrended, lf_band, normalize=False), lf_band))
        hf_power = float(np.trapezoid(
            lombscargle(beat_times, rr_detrended, hf_band, normalize=False), hf_band))
    except Exception:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan]

    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-12 else np.nan

    names.extend(feature_names)
    return [lf_power, hf_power, lf_hf_ratio]


def _hrv_nonlinear_features(r_peaks, fs, names):
    """Poincare-plot HRV: SD1 (beat-to-beat) and SD2 (longer-term) spread of RR[i] vs RR[i+1], plus their ratio."""
    prefix = "ecg"
    feature_names = [f"{prefix}_sd1", f"{prefix}_sd2", f"{prefix}_sd1_sd2_ratio"]

    if r_peaks is None or r_peaks.size < 5:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan]

    rr = np.diff(r_peaks) / fs
    diffs = np.diff(rr)
    sd1 = float(np.sqrt(np.var(diffs) / 2.0))
    sd2_sq = 2 * np.var(rr) - sd1 ** 2
    if sd2_sq <= 0:
        names.extend(feature_names)
        return [sd1, np.nan, np.nan]
    sd2 = float(np.sqrt(sd2_sq))
    ratio = sd1 / sd2 if sd2 > 1e-9 else np.nan

    names.extend(feature_names)
    return [sd1, sd2, float(ratio) if np.isfinite(ratio) else np.nan]


def _sample_entropy(x, m=2, r_rel=0.2):
    """Sample entropy (signal regularity/complexity) via pdist Chebyshev distance."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    r = r_rel * np.std(x)
    if r <= 0 or n < m + 2:
        return np.nan

    def _count_matches(m_):
        templates = np.array([x[i:i + m_] for i in range(n - m_ + 1)])
        d = pdist(templates, metric="chebyshev")
        return int(np.sum(d <= r))

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def _hrv_entropy_features(r_peaks, fs, names):
    """Sample entropy of the RR-interval series -- see _sample_entropy."""
    prefix = "ecg"
    feature_names = [f"{prefix}_sampen"]
    if r_peaks is None or r_peaks.size < 10:
        names.extend(feature_names)
        return [np.nan]
    rr = np.diff(r_peaks) / fs
    names.extend(feature_names)
    return [_sample_entropy(rr)]


def _qt_proxy_features(x_filled, r_peaks, fs, names):
    """R-to-T-peak interval as a cheap QT proxy, plus Fridericia-corrected QTc, QT dispersion, and T-wave amplitude/T-R ratio."""
    prefix = "ecg"
    feature_names = [f"{prefix}_qt_mean", f"{prefix}_qtc_mean", f"{prefix}_qt_std",
                      f"{prefix}_qt_range", f"{prefix}_t_amplitude_mean", f"{prefix}_t_r_ratio"]

    if x_filled is None or r_peaks is None or r_peaks.size < 5:
        names.extend(feature_names)
        return [np.nan] * 6

    rr = np.diff(r_peaks) / fs
    win_start = int(0.08 * fs)
    win_end = int(0.45 * fs)
    baseline_win = max(1, int(0.05 * fs))

    qts, qtcs, t_amps, tr_ratios = [], [], [], []
    for i, r in enumerate(r_peaks[:-1]):
        seg_start = r + win_start
        seg_end = min(r + win_end, x_filled.size, r_peaks[i + 1])
        if seg_end <= seg_start:
            continue
        segment = x_filled[seg_start:seg_end]
        t_peak_offset = np.argmax(segment)
        qt = (win_start + t_peak_offset) / fs
        qts.append(qt)
        if rr[i] > 0:
            qtcs.append(qt / (rr[i] ** (1 / 3)))

        base_start = max(0, r - baseline_win)
        if base_start >= r:
            continue
        baseline = np.mean(x_filled[base_start:r])
        r_amp = x_filled[r] - baseline
        t_amp = segment[t_peak_offset] - baseline
        t_amps.append(t_amp)
        if abs(r_amp) > 1e-6:
            tr_ratios.append(t_amp / r_amp)

    if len(qts) < 3:
        names.extend(feature_names)
        return [np.nan] * 6

    qt_mean = float(np.mean(qts))
    qtc_mean = float(np.mean(qtcs)) if qtcs else np.nan
    qt_std = float(np.std(qts))
    qt_range = float(np.max(qts) - np.min(qts))
    t_amp_mean = float(np.mean(t_amps)) if t_amps else np.nan
    tr_ratio = float(np.median(tr_ratios)) if tr_ratios else np.nan

    names.extend(feature_names)
    return [qt_mean, qtc_mean, qt_std, qt_range, t_amp_mean, tr_ratio]


def _eda_scr_features(x, fs, names, tonic_window_s=20, min_amp_rel=0.5):
    """Skin-conductance-response event features: count, amplitude, tonic slope, recency, habituation, sample entropy."""
    prefix = "eda"
    feature_names = [f"{prefix}_scr_count", f"{prefix}_scr_amp_mean", f"{prefix}_scr_amp_sum",
                      f"{prefix}_tonic_slope", f"{prefix}_time_since_last_scr",
                      f"{prefix}_scr_habituation", f"{prefix}_sampen"]

    x_filled = _interpolate_nans(x)
    if x_filled is None:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    window = max(1, int(tonic_window_s * fs))
    tonic = _moving_average(x_filled, window)
    phasic = x_filled - tonic

    tonic_idx = np.arange(tonic.size)
    tonic_slope = float(np.polyfit(tonic_idx, tonic, 1)[0])

    sampen = _sample_entropy(x_filled)

    min_distance = max(1, int(fs * 1.0))
    min_amp = max(1e-3, min_amp_rel * np.std(phasic))
    scr_peaks, _ = find_peaks(phasic, distance=min_distance, height=min_amp)

    scr_count = float(scr_peaks.size)
    if scr_peaks.size == 0:
        amp_mean, amp_sum = 0.0, 0.0
        time_since_last_scr = float(x_filled.size / fs)
        habituation = np.nan
    else:
        amps = phasic[scr_peaks]
        amp_mean = float(np.mean(amps))
        amp_sum = float(np.sum(amps))
        time_since_last_scr = float((x_filled.size - 1 - scr_peaks[-1]) / fs)
        if amps.size >= 3:
            habituation = float(np.polyfit(np.arange(amps.size), amps, 1)[0])
        else:
            habituation = np.nan

    names.extend(feature_names)
    return [scr_count, amp_mean, amp_sum, tonic_slope, time_since_last_scr, habituation, sampen]


def _bvp_peaks(x_filled, fs, min_rate_per_min=40, max_rate_per_min=200):
    """BVP pulse-peak detection, run once and reused by downstream BVP features."""
    if x_filled is None:
        return None
    peaks, _ = peak_intervals(x_filled, fs=fs,
                               min_rate_per_min=min_rate_per_min,
                               max_rate_per_min=max_rate_per_min,
                               prominence=0.5 * np.std(x_filled))
    return peaks


def _bvp_pulse_features(x_filled, peaks, fs, names, max_rate_per_min=200):
    """Pulse-rate variability (PRV), mean systolic rise time, and pulse amplitude."""
    prefix = "bvp"
    feature_names = [f"{prefix}_prv_sdnn", f"{prefix}_prv_rmssd",
                      f"{prefix}_rise_time_mean", f"{prefix}_pulse_amp_mean"]

    if x_filled is None or peaks is None:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan, np.nan]

    intervals = np.diff(peaks) / fs
    if intervals.size < 2:
        names.extend(feature_names)
        return [np.nan, np.nan, np.nan, np.nan]

    prv_sdnn = float(np.std(intervals))
    prv_rmssd = float(np.sqrt(np.mean(np.diff(intervals) ** 2)))

    min_distance = max(1, int(fs * 60 / max_rate_per_min))
    prominence = 0.5 * np.std(x_filled)
    troughs, _ = find_peaks(-x_filled, distance=min_distance, prominence=prominence)
    rise_times, amps = [], []
    for p in peaks:
        prior_troughs = troughs[troughs < p]
        if prior_troughs.size:
            t = prior_troughs[-1]
            rise_times.append((p - t) / fs)
            amps.append(x_filled[p] - x_filled[t])

    rise_time_mean = float(np.mean(rise_times)) if rise_times else np.nan
    pulse_amp_mean = float(np.mean(amps)) if amps else np.nan

    names.extend(feature_names)
    return [prv_sdnn, prv_rmssd, rise_time_mean, pulse_amp_mean]


def _bvp_augmentation_index(x_filled, peaks, names):
    """PPG augmentation index: median ratio of the reflected-wave peak to the main pulse peak."""
    prefix = "bvp"
    feature_names = [f"{prefix}_augmentation_index"]

    if x_filled is None or peaks is None or peaks.size < 3:
        names.extend(feature_names)
        return [np.nan]

    ratios = []
    for i in range(len(peaks) - 1):
        p = peaks[i]
        next_p = peaks[i + 1]
        early_height = x_filled[p]
        search_end = p + int(0.6 * (next_p - p))
        if search_end <= p + 2 or search_end > x_filled.size:
            continue
        segment = x_filled[p:search_end]
        sub_peaks, _ = find_peaks(segment)
        if sub_peaks.size == 0 or early_height <= 1e-6:
            continue
        late_height = float(np.max(segment[sub_peaks]))
        ratios.append(late_height / early_height)

    if len(ratios) < 2:
        names.extend(feature_names)
        return [np.nan]

    names.extend(feature_names)
    return [float(np.median(ratios))]


def _ptt_features(ecg_r_peaks, bvp_peaks, fs_ecg, fs_bvp, names):
    """Pulse transit time: delay between each ECG R-peak and the BVP peak it produces peripherally."""
    prefix = "ptt"
    feature_names = [f"{prefix}_mean", f"{prefix}_std"]

    if (ecg_r_peaks is None or bvp_peaks is None
            or ecg_r_peaks.size < 2 or bvp_peaks.size < 2):
        names.extend(feature_names)
        return [np.nan, np.nan]

    ecg_times = ecg_r_peaks / fs_ecg
    bvp_times = bvp_peaks / fs_bvp

    ptts = []
    for t_r in ecg_times:
        later = bvp_times[bvp_times > t_r]
        if later.size:
            gap = later[0] - t_r
            if gap < 1.0:
                ptts.append(gap)

    if len(ptts) < 2:
        names.extend(feature_names)
        return [np.nan, np.nan]

    names.extend(feature_names)
    return [float(np.mean(ptts)), float(np.std(ptts))]


def _wavelet_energy_features(x_filled, signal_size, prefix, names, wavelet="db4", max_level=6):
    """Multi-scale wavelet-decomposition energy fractions (finest l1 to coarsest l{level})."""
    level = min(max_level, pywt.dwt_max_level(signal_size, pywt.Wavelet(wavelet).dec_len))
    feature_names = [f"{prefix}_wavelet_energy_l{i}" for i in range(1, level + 1)]

    if x_filled is None:
        names.extend(feature_names)
        return [np.nan] * level

    coeffs = pywt.wavedec(x_filled, wavelet, level=level)
    # coeffs = [cA_level, cD_level, ..., cD_1]; drop the approximation and
    # reverse so output order is finest (l1) -> coarsest.
    detail_energies = np.array([np.sum(c ** 2) for c in coeffs[1:]])[::-1]
    total = float(np.sum(detail_energies))
    if total <= 0:
        names.extend(feature_names)
        return [np.nan] * level

    names.extend(feature_names)
    return [float(e / total) for e in detail_energies]


def make_features_for_window(record, feature_names_out):
    """Builds the feature vector for one window; feature_names_out is populated on the first call and checked for consistency after."""
    names = []
    vals = []
    eps = 1e-6  # guards divide-by-zero in the motion-quality ratios below

    # --- BVP (E4 PPG) ---
    bvp = record["e4_bvp"]
    bvp_filled = _interpolate_nans(bvp)
    bvp_peaks = _bvp_peaks(bvp_filled, fs=64)
    bvp_stats = _stats(bvp, "bvp", names)
    vals += bvp_stats
    vals += _slope(bvp, "bvp", names)
    vals += _bvp_pulse_features(bvp_filled, bvp_peaks, fs=64, names=names)
    vals += _wavelet_energy_features(bvp_filled, 19200, "bvp", names)
    vals += _bvp_augmentation_index(bvp_filled, bvp_peaks, names)

    # --- E4 HR (device-derived heart rate trace) ---
    hr = record["e4_hr"]
    vals += _stats(hr, "e4_hr", names)
    vals += _block_trend_features(hr, "e4_hr", names)

    # --- EDA ---
    eda = record["e4_eda"]
    vals += _stats(eda, "eda", names)
    vals += _slope(eda, "eda", names)
    vals += _eda_scr_features(eda, fs=4, names=names)
    vals += _block_trend_features(eda, "eda", names)
    eda_filled = _interpolate_nans(eda)
    vals += _wavelet_energy_features(eda_filled, 1200, "eda", names)

    # --- Temperature (detrended: raw skin temp is a stable per-person
    # baseline trait, otherwise a subject-identity leak) ---
    temp = record["e4_temp"]
    temp_filled = _interpolate_nans(temp)
    vals += _stats_detrended(temp_filled, fs=4, prefix="temp", names=names)
    vals += _slope(temp, "temp", names)
    vals += _block_trend_features(temp, "temp", names)

    # --- E4 accelerometer: squared magnitude a_sq(t) = ax^2+ay^2+az^2 ---
    acc = record["e4_acc"]
    acc_sq = np.nansum(acc.astype(np.float64) ** 2, axis=1)
    acc_sq_stats = _stats(acc_sq, "e4_acc_sq", names)
    vals += acc_sq_stats

    # BVP signal-quality proxy: pulse-amplitude variability vs. wrist motion energy.
    bvp_motion_quality = float(bvp_stats[1] / (acc_sq_stats[0] + eps))
    names.append("bvp_motion_quality")
    vals.append(bvp_motion_quality)

    # --- Zephyr ECG ---
    ecg = record["zephyr_ecg"]
    ecg_filled = _interpolate_nans(ecg)
    ecg_stats = _stats_detrended(ecg_filled, fs=250, prefix="ecg", names=names)
    vals += ecg_stats
    ecg_r_peaks = _ecg_r_peaks_nk(ecg_filled, fs=250)
    vals += _hrv_features(ecg_r_peaks, fs=250, names=names)
    vals += _hrv_freq_features(ecg_r_peaks, fs=250, names=names)
    vals += _hrv_nonlinear_features(ecg_r_peaks, fs=250, names=names)
    vals += _qt_proxy_features(ecg_filled, ecg_r_peaks, fs=250, names=names)
    vals += _hrv_entropy_features(ecg_r_peaks, fs=250, names=names)
    vals += _ptt_features(ecg_r_peaks, bvp_peaks, fs_ecg=250, fs_bvp=64, names=names)

    # --- Zephyr accelerometer magnitude ---
    zacc = record["zephyr_acc"]
    zacc_sq = np.nansum(zacc.astype(np.float64) ** 2, axis=1)
    zacc_sq_stats = _stats(zacc_sq, "zephyr_acc_sq", names)
    vals += zacc_sq_stats

    # ECG signal-quality proxy, same rationale as bvp_motion_quality above.
    ecg_motion_quality = float(ecg_stats[1] / (zacc_sq_stats[0] + eps))
    names.append("ecg_motion_quality")
    vals.append(ecg_motion_quality)

    # --- Zephyr breathing ---
    breathing = record["zephyr_breathing"]
    breathing_filled = _interpolate_nans(breathing)
    vals += _stats_detrended(breathing_filled, fs=25, prefix="breathing", names=names)
    breathing_rate_vals = _breathing_rate_features(breathing_filled, fs=25, names=names)
    vals += breathing_rate_vals
    vals += _block_trend_features(breathing, "breathing", names)

    if not feature_names_out:
        feature_names_out.extend(names)
    else:
        assert feature_names_out == names, "Feature name/order mismatch between windows"

    return np.array(vals, dtype=np.float64)


def iter_participant_files(data_dir):
    return sorted(Path(data_dir).glob("*.npz"))


def build_feature_matrix(data_dir, has_target):
    """Processes one .npz file at a time; returns (Z, y, feature_names), y is None if has_target is False."""
    feature_names = []
    feature_rows = []
    targets = [] if has_target else None

    for path in iter_participant_files(data_dir):
        with np.load(path, allow_pickle=False) as data:
            n = data["e4_bvp"].shape[0]
            fields = {
                "e4_bvp": data["e4_bvp"],
                "e4_hr": data["e4_hr"],
                "e4_eda": data["e4_eda"],
                "e4_temp": data["e4_temp"],
                "e4_acc": data["e4_acc"],
                "zephyr_ecg": data["zephyr_ecg"],
                "zephyr_acc": data["zephyr_acc"],
                "zephyr_breathing": data["zephyr_breathing"],
            }

            if has_target:
                glucose = data["glucose"]

            for i in range(n):
                record = {k: v[i] for k, v in fields.items()}
                row = make_features_for_window(record, feature_names)
                feature_rows.append(row)

            if has_target:
                targets.append(glucose)

    Z = np.vstack(feature_rows)
    y = np.concatenate(targets) if has_target else None
    return Z, y, feature_names


def cmd_train(protocol, train_dir, model_path):
    Z, y, feature_names = build_feature_matrix(train_dir, has_target=True)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Z_imputed = imputer.fit_transform(Z)
    assert Z_imputed.shape[1] == len(feature_names), (
        f"Imputer changed feature count: {len(feature_names)} -> {Z_imputed.shape[1]}"
    )

    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z_imputed)

    model = LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, max_iter=50000, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Z_scaled, y)

    # y_hat = intercept_s + coef_s . ((z - mean) / scale)
    #       = (intercept_s - sum(coef_s * mean / scale)) + sum((coef_s/scale) * z)
    coef_scaled = model.coef_
    coef_raw = coef_scaled / scaler.scale_
    intercept_raw = model.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)

    state = {
        "format_version": FORMAT_VERSION,
        "protocol": protocol,
        "intercept": float(intercept_raw),
        "coef": coef_raw.astype(np.float64),
        "feature_names": feature_names,
        "preprocessing_state": {
            "imputer": imputer,
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(state, f)


def cmd_feature_engineering(protocol, test_dir, model_path, output_path):
    with open(model_path, "rb") as f:
        state = pickle.load(f)

    assert state["protocol"] == protocol, "Model/protocol mismatch"

    Z, _, feature_names = build_feature_matrix(test_dir, has_target=False)
    assert feature_names == state["feature_names"], "Feature name/order mismatch vs. training"

    imputer = state["preprocessing_state"]["imputer"]
    Z_imputed = imputer.transform(Z)

    assert Z_imputed.shape[1] == len(state["feature_names"]), (
        f"Imputer output width {Z_imputed.shape[1]} != "
        f"len(feature_names) {len(state['feature_names'])}"
    )
    assert np.isfinite(Z_imputed).all(), "Non-finite values remain in final feature matrix"

    np.save(output_path, Z_imputed)


def main():
    usage = (
        "usage:\n"
        "  part_d.py train <protocol> <train_dir> <model_path>\n"
        "  part_d.py feature_engineering <protocol> <test_dir> <model_path> <output_path>"
    )
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        _, _, protocol, train_dir, model_path = sys.argv
        cmd_train(protocol, train_dir, model_path)
    elif mode == "feature_engineering":
        _, _, protocol, test_dir, model_path, output_path = sys.argv
        cmd_feature_engineering(protocol, test_dir, model_path, output_path)
    else:
        print(usage)
        sys.exit(1)


if __name__ == "__main__":
    main()

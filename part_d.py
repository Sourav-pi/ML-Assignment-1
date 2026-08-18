import numpy as np
import pandas as pd
import pickle
import os
import sys
import warnings
import time
import itertools
from pathlib import Path

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression, HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scipy.signal import find_peaks, lombscargle, welch
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist

import neurokit2 as nk
import pywt

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

FORMAT_VERSION = 1
DEBUG = False

def _stats(x, prefix, names):
    """
    Basic summary stats for a 1D array, ignoring NaNs. Returns values + appends names.
    If a window has no valid samples at all for this modality, emits np.nan rather
    than a hardcoded fallback (e.g. 0.0) — a fabricated constant would silently
    distort downstream model fitting/interpretation, whereas np.nan is handled
    explicitly and only ever imputed using train-set statistics (see cmd_train).
    """
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
    """
    Returns peak indices and inter-peak intervals (seconds).
    `prominence` rejects peaks that don't rise clearly above the local
    noise floor -- without it, `distance` alone lets small noise wiggles
    (e.g. T-waves, sensor noise) count as full peaks and silently double
    (or more) the detected event rate.
    """
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
    """Linear interpolation over isolated missing samples. Returns None if
    too much of the signal is missing to interpolate meaningfully."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    if valid.sum() < 0.5 * valid.size:
        return None
    idx = np.arange(x.size)
    return np.interp(idx, idx[valid], x[valid])


def _breathing_rate_features(x_filled, fs, names, min_rate_per_min=6, max_rate_per_min=40):
    """
    Respiration-rate features from the raw breathing waveform: instantaneous
    breathing rate (from peak-to-peak timing) and breath amplitude, both
    mean/std across the window.

    Takes an already-interpolated signal (see _interpolate_nans) rather than
    interpolating internally, since the caller (make_features_for_window)
    also needs the same interpolated array for _stats_detrended -- computing
    it once and sharing it avoids running np.interp over the same 7500-sample
    breathing signal twice per window.

    NaN-safe: if x_filled is None (too much of the window was missing to
    interpolate meaningfully), or too few breaths are detected to form an
    interval, emits np.nan so the downstream imputer (fit on train only)
    handles it consistently with every other feature.
    """
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

    inst_rate = 60.0 / intervals  # breaths per minute, per interval
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
    """
    Edge-corrected moving average via scipy.ndimage.uniform_filter1d, which
    computes this in O(N) via a running sum -- unlike np.convolve, which is
    O(N * window). That distinction matters a lot here: ECG is 75000
    samples at a 30s*250Hz=7500-sample window, so a naive convolve does
    ~560M multiply-adds *per window*, dwarfing every other feature
    computation (including NeuroKit2) once multiplied across ~15-18k
    training windows. mode="nearest" repeats the edge value outward,
    avoiding the implicit zero-padding that mode="same" convolution would
    use -- critical when the raw signal has a large nonzero baseline (e.g.
    raw ECG counts ~2,000,000): zero-padding would collapse the baseline
    estimate toward zero right at the edges, leaving a huge residual there.
    """
    if window <= 1:
        return x.copy()
    return uniform_filter1d(x, size=window, mode="nearest")


def _detrend(x_filled, fs, baseline_window_s=30):
    """
    Removes a slow moving-average baseline from an already-interpolated
    signal (see _interpolate_nans), isolating faster within-window signal
    shape from the sensor's arbitrary calibration offset/drift. The raw
    zephyr_ecg/zephyr_breathing channels here are unscaled device counts
    with large, subject-specific DC offsets (e.g. breathing_mean ~
    4,100,000) that reflect electrode contact/strap fit -- not physiology --
    and were found to correlate with glucose only because they act as a
    subject fingerprint (see the d1 vs d2 correlation comparison).
    Detrending removes that offset before any level-based stat is computed
    on these two channels.

    Takes a pre-filled array rather than interpolating internally, so
    callers that also need the interpolated signal elsewhere (e.g.
    _ecg_r_peaks_nk) can compute it once and share it.
    """
    if x_filled is None:
        return None
    window = max(1, int(baseline_window_s * fs))
    if window >= x_filled.size:
        return x_filled - np.mean(x_filled)
    baseline = _moving_average(x_filled, window)
    return x_filled - baseline


def _stats_detrended(x_filled, fs, prefix, names, baseline_window_s=30):
    """Like _stats, but computed on the detrended signal (see _detrend) so
    mean/min/max/range reflect real within-window signal shape rather than
    the sensor's arbitrary calibration offset. Takes an already-interpolated
    array (see _interpolate_nans) rather than interpolating internally."""
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
    """
    Splits the window into n_blocks equal chunks and reports how much the
    signal moved from the first half of the window to the second half
    (block_trend), plus how much block-to-block variability there is
    (block_var). Complements the whole-window linear _slope: a window that
    dips then recovers has ~zero net slope but real block_var, which a
    single slope value can't distinguish from a flat window -- and acute
    autonomic responses are often about *when* something changes, not just
    the net drift over 5 minutes.
    """
    feature_names = [f"{prefix}_block_trend", f"{prefix}_block_var"]
    x_filled = _interpolate_nans(x)
    if x_filled is None:
        names.extend(feature_names)
        return [np.nan, np.nan]

    blocks = np.array_split(x_filled, n_blocks)
    block_means = np.array([np.mean(b) for b in blocks])

    half = n_blocks // 2
    block_trend = float(np.mean(block_means[half:]) - np.mean(block_means[:half]))
    block_var = float(np.std(block_means))

    names.extend(feature_names)
    return [block_trend, block_var]


def _ecg_r_peaks_nk(x_filled, fs):
    """
    Runs NeuroKit2's validated R-peak detector once per window. The result
    is reused by both _hrv_features and _ptt_features below instead of
    each independently re-running peak detection on the same 75000-sample
    ECG signal -- detection is the expensive, hard-to-replicate part of
    NeuroKit2's pipeline, worth computing exactly once.

    Takes an already-interpolated signal (see _interpolate_nans) so the
    caller can share it with _stats_detrended rather than interpolating the
    same 75000-sample ECG array twice.

    Returns R-peak sample indices, or None if detection fails / finds too
    few peaks to be usable. Wrapped in a try/except: NeuroKit2 can raise on
    very short/degenerate windows it can't find a usable QRS complex in.
    """
    if x_filled is None:
        return None
    try:
        with warnings.catch_warnings():
            # NeuroKit2 can internally call np.nanmean/np.nanvar on an
            # empty/near-empty RR-interval array for degenerate windows,
            # which warns ("Mean of empty slice" etc.) -- exactly the case
            # this function already returns None for. Expected noise, not
            # a real problem.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _, info = nk.ecg_peaks(x_filled, sampling_rate=fs)
        r_peaks = np.asarray(info["ECG_R_Peaks"])
    except Exception:
        return None
    if r_peaks.size < 2:
        return None
    return r_peaks


def _hrv_features(r_peaks, fs, names):
    """
    Heart-rate variability features computed directly from already-detected
    R-peak sample indices (see _ecg_r_peaks_nk): R-peak-derived heart rate,
    SDNN (overall RR-interval variability) and RMSSD (beat-to-beat
    variability). RMSSD in particular is sensitive to sympathetic/
    parasympathetic shifts -- the same autonomic response that acute
    glucose swings (especially hypoglycemia) trigger -- unlike plain
    amplitude stats on the raw ECG trace.

    Deliberately does NOT call nk.hrv_time: that computes a much wider
    suite of time/frequency/nonlinear HRV metrics we don't use, plus
    pandas DataFrame construction, on every single window -- by far the
    most expensive step in the original pipeline. Keeping only NeuroKit2's
    peak *detection* (the validated, hard-to-replicate part) and computing
    these 3 specific metrics with plain numpy gives the same values for a
    fraction of the cost.
    """
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
    """
    Frequency-domain HRV: power in the low-frequency (LF, 0.04-0.15 Hz --
    baroreflex/sympathetic-linked) and high-frequency (HF, 0.15-0.4 Hz --
    respiratory-linked/parasympathetic) bands, and their ratio (LF/HF), the
    standard clinical marker of sympathovagal balance. Complements the
    time-domain SDNN/RMSSD in _hrv_features with the specific autonomic-
    balance signal that acute glucose swings are expected to shift, and
    that time-domain variability alone can't distinguish (two windows with
    identical SDNN can have very different LF/HF).

    Reuses r_peaks from _ecg_r_peaks_nk (no extra peak detection). Uses the
    Lomb-Scargle periodogram directly on the irregularly-spaced RR-interval
    series (beat times, not a fixed sample rate) rather than resampling to
    a uniform grid first -- the standard approach for HRV spectral analysis,
    since RR intervals are inherently unevenly spaced in time. A 300s window
    is actually the classic recommended duration for short-term HRV
    frequency analysis, so this fits well here.

    min_beats=20 is a conservative floor (literature typically wants 200+
    beats for a reliable 5-minute HRV spectrum) below which the estimate is
    considered too noisy to trust; emits np.nan in that case.
    """
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
    """
    Poincare-plot nonlinear HRV: SD1 (short-term/beat-to-beat variability,
    perpendicular to the line of identity -- closely related to RMSSD/
    parasympathetic tone) and SD2 (longer-term variability, along the line
    of identity -- closer to overall SDNN-style variability), plus their
    ratio. A geometric/nonlinear lens on RR-interval structure, distinct
    from both the linear time-domain metrics in _hrv_features and the
    spectral ones in _hrv_freq_features -- captures beat-to-beat structure
    those miss. Reuses r_peaks from _ecg_r_peaks_nk, no extra peak
    detection. Formula validated against NeuroKit2's own hrv_nonlinear on
    synthetic RR data before deploying (matched to ~4 decimal places).
    """
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
    """
    Sample entropy: quantifies signal regularity/complexity. Lower values
    mean more self-similar/predictable, higher values mean more irregular.
    Reduced HRV sample entropy is a documented marker of diabetic cardiac
    autonomic neuropathy (Cardiovascular autonomic function analysis using
    approximate entropy from 24-h HRV in type 2 diabetes -- PMC4364858),
    and entropy analysis more broadly is used for autonomic/complexity
    assessment in aging and diabetes (Frontiers in Physiology, 2026,
    10.3389/fphys.2026.1795118). Applied here to both ECG RR-intervals and
    the raw EDA signal.

    Uses scipy.spatial.distance.pdist (Chebyshev/max-norm distance,
    matching the standard sample-entropy definition) instead of a naive
    O(N^2) Python loop -- ~9x faster in practice (verified: 35ms -> 4ms for
    a 1200-sample EDA window), which matters given this runs on every
    window across ~18000 windows per protocol.
    """
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
    """
    R-to-T-peak interval: a computationally cheap proxy for the clinical
    QT interval (which technically requires QRS-onset-to-T-wave-OFFSET via
    full waveform delineation -- NeuroKit2's ecg_delineate() was benchmarked
    at ~2.2s/window, ~170x slower than this entire feature pipeline
    combined, which would take ~11 hours across d3's ~18000 windows alone;
    infeasible at this scale). This simplified proxy searches a fixed
    physiological window (80-450ms) after each R-peak for the T-wave's
    peak, which is a standard, much cheaper approximation used in
    wearable/ambulatory ECG work when full delineation isn't tractable.
    Validated on synthetic ECG with a known injected ~300ms QT: recovered
    296ms (1.4% error), at 0.5ms/window (~4500x cheaper than delineation).

    Scientifically motivated by hypoglycemia-induced QT/QTc prolongation
    specifically documented in Type 1 diabetics -- the population in this
    dataset -- including the "dead-in-bed syndrome" literature (Diabetes
    Care 2006, 10.2337/diacare.29.2.427; Diabetologia 2010,
    10.1007/s00125-010-1802-0). QTc uses Fridericia's correction
    (QT / RR^(1/3)), reported as more accurate than Bazett's for
    hypoglycemia-related QT changes specifically, and generally the more
    accurate/heart-rate-independent correction overall, FDA-preferred for
    QT submissions (Diabetologia 2010; comparison of QTc formulae across
    heart rate ranges, PMC12811695).

    Also reports, from the same per-beat loop (no extra detection cost):
    - qt_range: beat-to-beat QT dispersion (max-min across the window's
      beats). QT dispersion is documented to specifically INCREASE during
      hypoglycemia in diabetes (10475998; PMC5014905) -- a distinct,
      outlier-sensitive complement to qt_std's variance-based estimate.
      (Note: classical clinical "QT dispersion" is measured across
      multiple simultaneous ECG leads; this is a single-lead, across-time
      analog -- beat-to-beat QT variability within one lead, not the same
      construct, but capturing a related instability signal.)
    - t_amplitude_mean, t_r_ratio: T-wave amplitude and its ratio to the
      R-wave amplitude, both measured relative to each beat's own local
      isoelectric baseline (~50ms just before R) rather than the raw
      signal value -- necessary because the raw ECG still carries a large
      per-subject calibration offset even after _stats_detrended's
      window-level baseline removal (per-beat is finer-grained). The T/R
      *ratio* specifically also cancels a per-subject amplifier GAIN
      factor that a plain amplitude difference would not, which raw ECG
      amplitude features here were suspected of still carrying (see the
      residual-multiplicative-leakage note from ecg_min/max/range/std
      analysis). T-wave flattening (decreasing amplitude) during
      hypoglycemia is well documented, catecholamine-mediated (epinephrine
      response correlates directly with T-wave flattening), in Type 1
      diabetics specifically (Diabetologia 2007, 10.1007/s00125-007-0902-y;
      PMC6931981). t_r_ratio uses the MEDIAN across beats, not the mean:
      caught via synthetic testing that a plain mean lets a handful of
      noisy, near-zero-denominator beats (small r_amp) dominate and produce
      an implausible aggregate ratio -- same reasoning already applied to
      _bvp_augmentation_index's ratio.
    """
    prefix = "ecg"
    feature_names = [f"{prefix}_qt_mean", f"{prefix}_qtc_mean", f"{prefix}_qt_std",
                      f"{prefix}_qt_range", f"{prefix}_t_amplitude_mean", f"{prefix}_t_r_ratio"]

    if x_filled is None or r_peaks is None or r_peaks.size < 5:
        names.extend(feature_names)
        return [np.nan] * 6

    rr = np.diff(r_peaks) / fs
    win_start = int(0.08 * fs)
    win_end = int(0.45 * fs)
    baseline_win = max(1, int(0.05 * fs))  # ~50ms isoelectric baseline just before R

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


def _hrv_rsa_power(r_peaks, fs_ecg, resp_rate_bpm, names):
    """
    RR-interval spectral power at the window's own measured respiratory
    frequency (from _breathing_rate_features) -- a targeted respiratory
    sinus arrhythmia (RSA) measure, distinct from _hrv_freq_features'
    generic 0.15-0.4Hz HF band: this centers exactly on each window's
    actual breathing rate rather than a fixed population-average band.
    RSA magnitude directly indexes parasympathetic/vagal tone (Respiratory
    Sinus Arrhythmia Mechanisms in Young Obese Subjects, PMC7079685), and
    diabetic autonomic neuropathy is associated with reduced RSA.

    Reuses r_peaks from _ecg_r_peaks_nk; called from the breathing section
    of make_features_for_window since it needs breathing_resp_rate, which
    isn't computed until then.
    """
    prefix = "ecg"
    feature_names = [f"{prefix}_rsa_power"]

    if (r_peaks is None or r_peaks.size < 20 or resp_rate_bpm is None
            or not np.isfinite(resp_rate_bpm) or resp_rate_bpm <= 0):
        names.extend(feature_names)
        return [np.nan]

    beat_times = r_peaks[1:] / fs_ecg
    rr = np.diff(r_peaks) / fs_ecg
    rr_detrended = rr - np.mean(rr)

    target_hz = resp_rate_bpm / 60.0
    freq_band = np.linspace(max(0.01, target_hz - 0.02), target_hz + 0.02, 10) * 2 * np.pi

    try:
        power = float(np.trapezoid(
            lombscargle(beat_times, rr_detrended, freq_band, normalize=False), freq_band))
    except Exception:
        names.extend(feature_names)
        return [np.nan]

    names.extend(feature_names)
    return [power]


def _eda_scr_features(x, fs, names, tonic_window_s=20, min_amp_rel=0.5):
    """
    Skin-conductance-response (SCR) event features from EDA: how many
    sudden phasic spikes occurred in the window, and how large they were.
    Sweating (adrenergic sweat-gland activation) is a textbook acute
    hypoglycemia symptom, so SCR *events* -- not the mean EDA level -- are
    the physiologically motivated signal here.

    Tonic (slow baseline) component is estimated with a moving average and
    subtracted to isolate the phasic (fast) component, in which peaks are
    detected as SCR events. NaN-safe like the other peak-based features.

    Also reports the tonic component's own within-window slope (drift in
    overall arousal level, distinct from the mean-EDA-level feature in
    _stats -- slope is shift-invariant, so it isolates trend from absolute
    level -- and smoother/less spike-sensitive than fitting a line straight
    through the raw noisy signal) and the time since the most recent SCR
    event, relative to the end of the window (a recency-weighted signal:
    "how long ago was the last arousal event" is plausibly more relevant to
    the glucose reading at the end of the window than the total count over
    the full 5 minutes). Windows with zero SCR events get the full window
    duration here (a real, informative value -- "no recent event" -- rather
    than np.nan, which would just be imputed away).

    Also reports SCR habituation (the slope of successive SCR amplitudes
    in temporal order -- a negative slope means the sympathetic response
    is diminishing across repeated events within the window, a classic
    psychophysiological adaptation/habituation pattern) and the raw EDA
    signal's sample entropy (see _sample_entropy), a complexity marker
    distinct from any single summary statistic.
    """
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

    # Relative threshold: EDA's absolute scale varies hugely by person/device
    # calibration, so a fixed absolute height cuts off very differently across
    # subjects. Scale to this window's own phasic variability instead, with a
    # tiny floor so a near-flat phasic signal doesn't trigger on pure noise.
    min_distance = max(1, int(fs * 1.0))  # SCRs don't repeat faster than ~1/s
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
    """
    Detects BVP pulse peaks once per window. The result is reused by both
    _bvp_pulse_features and _ptt_features below instead of each
    independently re-running peak detection (same signal, same parameters)
    on the same 19200-sample BVP array.

    Returns peak sample indices (possibly empty/short), or None only if
    x_filled itself is None.
    """
    if x_filled is None:
        return None
    peaks, _ = peak_intervals(x_filled, fs=fs,
                               min_rate_per_min=min_rate_per_min,
                               max_rate_per_min=max_rate_per_min,
                               prominence=0.5 * np.std(x_filled))
    return peaks


def _bvp_pulse_features(x_filled, peaks, fs, names, max_rate_per_min=200):
    """
    Pulse-morphology features from BVP: pulse-rate variability (PRV, the
    PPG analogue of HRV), mean systolic rise time, and pulse amplitude.
    Vascular tone/blood viscosity shifts (glucose- and adrenaline-linked)
    subtly change how sharply and how strongly each pulse rises, which
    plain waveform stats don't capture.

    Takes an already-interpolated signal and already-detected peaks (see
    _bvp_peaks) rather than recomputing either internally -- both are
    shared with _ptt_features.

    NaN-safe like the other peak-based features.
    """
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
    """
    PPG augmentation index: the ratio of the late systolic peak (the
    reflected-wave peak/inflection following the main pulse peak) to the
    early systolic peak (the main pulse peak itself) -- an established
    arterial-stiffness/vascular-tone marker (Diastolic Augmentation Index
    Improves Radial Augmentation Index in Assessing Arterial Stiffness,
    PMC5517606; Arterial stiffness assessment using PPG feature extraction,
    Scientific Reports 10.1038/s41598-024-51395-y). Distinct from the
    existing rise-time/pulse-amplitude features -- this specifically
    targets the reflected-wave component, not the primary upstroke.

    For each beat, searches the region after the main peak (up to ~60% of
    the way to the next beat) for a secondary local maximum. Reports the
    median late/early ratio across beats (median, not mean, since a few
    beats with a poorly-defined reflected wave -- no secondary peak found
    -- are simply skipped rather than corrupting the estimate).
    """
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


def _teager_energy_features(x_filled, prefix, names):
    """
    Teager-Kaiser energy operator: Psi(x[n]) = x[n]^2 - x[n-1]*x[n+1], a
    nonlinear operator jointly sensitive to a signal's instantaneous
    amplitude and frequency, cited directly as one of the features Monte-
    Moreno (2011) -- the foundational PPG-based non-invasive glucose
    estimation reference cited in this assignment's PDF -- used for
    PPG-based glucose estimation. Reports the mean/std of the resulting
    energy time series across the window.
    """
    feature_names = [f"{prefix}_teager_energy_mean", f"{prefix}_teager_energy_std"]

    if x_filled is None or x_filled.size < 3:
        names.extend(feature_names)
        return [np.nan, np.nan]

    teager = x_filled[1:-1] ** 2 - x_filled[:-2] * x_filled[2:]

    names.extend(feature_names)
    return [float(np.mean(teager)), float(np.std(teager))]


def _spectral_entropy_features(x_filled, fs, prefix, names, nperseg=256):
    """
    Spectral entropy: Shannon entropy of the normalized power spectral
    density (via Welch's method -- BVP is regularly sampled, unlike the
    RR-interval series elsewhere, so Welch is the standard choice here
    rather than Lomb-Scargle), normalized to [0, 1]. Another feature
    category Monte-Moreno (2011) used directly for PPG-based glucose
    estimation ("spectral entropy statistics"). Low values indicate a
    signal dominated by a few frequency components (regular/periodic);
    high values indicate energy spread broadly across frequencies
    (irregular/noisy).
    """
    feature_names = [f"{prefix}_spectral_entropy"]

    if x_filled is None:
        names.extend(feature_names)
        return [np.nan]

    try:
        _, psd = welch(x_filled, fs=fs, nperseg=min(nperseg, x_filled.size))
        psd_norm = psd / (np.sum(psd) + 1e-12)
        psd_norm = psd_norm[psd_norm > 0]
        if psd_norm.size < 2:
            names.extend(feature_names)
            return [np.nan]
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        entropy_norm = float(entropy / np.log2(psd_norm.size))
    except Exception:
        names.extend(feature_names)
        return [np.nan]

    names.extend(feature_names)
    return [entropy_norm]


def _ptt_features(ecg_r_peaks, bvp_peaks, fs_ecg, fs_bvp, names):
    """
    Pulse transit time (PTT): the delay between each ECG R-peak (electrical
    onset of a heartbeat) and the BVP peak it produces at the periphery.
    PTT is a validated proxy for arterial stiffness/vascular tone -- a more
    direct mechanistic link to glucose- and adrenaline-driven vascular
    effects than either signal's HRV/PRV alone, since it directly measures
    how fast the pulse wave travels rather than just how regularly it
    repeats.

    Reuses ecg_r_peaks (from _ecg_r_peaks_nk) and bvp_peaks (from
    _bvp_peaks) instead of re-running peak detection on either signal a
    second time.

    NaN-safe like the other peak-based features. Implausible gaps (>1s,
    suggesting a missed/mismatched beat) are discarded per-pair rather than
    corrupting the mean.
    """
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
            if gap < 1.0:  # reject implausible pairings (missed beat etc.)
                ptts.append(gap)

    if len(ptts) < 2:
        names.extend(feature_names)
        return [np.nan, np.nan]

    names.extend(feature_names)
    return [float(np.mean(ptts)), float(np.std(ptts))]


def _wavelet_energy_features(x_filled, signal_size, prefix, names, wavelet="db4", max_level=6):
    """
    Multi-scale energy distribution via discrete wavelet transform (using
    PyWavelets, explicitly permitted for Part (d)): the fraction of total
    signal energy at each detail level, from finest/highest-frequency (l1)
    to coarsest/lowest-frequency (l{level}). A complementary, non-Fourier
    lens on signal content vs. the peak-based features elsewhere -- e.g.
    captures how "sharp vs. smooth" the BVP waveform is across time-scales,
    or how EDA's fast/slow dynamics are distributed, without needing
    explicit peak detection. Applied to BVP and EDA specifically -- the two
    modalities without an existing frequency-domain feature (ECG already
    has _hrv_freq_features).

    Reports energy *fractions* (each level's energy / total energy across
    all requested levels), not raw energy -- invariant to a constant
    multiplicative amplitude scale, so this doesn't introduce a new
    subject-calibration-scale leakage path the way raw energy would.

    signal_size is passed explicitly (the fixed array length for this
    modality, e.g. 19200 for BVP) rather than derived from x_filled, so the
    decomposition level -- and therefore feature_names -- is identical
    across every window, including degenerate ones where x_filled is None.
    """
    level = min(max_level, pywt.dwt_max_level(signal_size, pywt.Wavelet(wavelet).dec_len))
    feature_names = [f"{prefix}_wavelet_energy_l{i}" for i in range(1, level + 1)]

    if x_filled is None:
        names.extend(feature_names)
        return [np.nan] * level

    coeffs = pywt.wavedec(x_filled, wavelet, level=level)
    # coeffs = [cA_level, cD_level, cD_(level-1), ..., cD_1] (approximation
    # first, then details coarsest -> finest). Drop the approximation
    # (mostly the slow/DC component already captured by _stats(_detrended)
    # elsewhere) and reverse so output order is finest (l1) -> coarsest.
    detail_energies = np.array([np.sum(c ** 2) for c in coeffs[1:]])[::-1]
    total = float(np.sum(detail_energies))
    if total <= 0:
        names.extend(feature_names)
        return [np.nan] * level

    names.extend(feature_names)
    return [float(e / total) for e in detail_energies]


# Features that consistently rank at or near the top of the
# feature-glucose correlation table across all three protocols (d1/d2/d3,
# observed repeatedly via `make diagnose-dX`) -- ECG timing/HRV features
# and breathing variability dominate throughout. Pairwise products among
# just this set are far more likely to surface a genuine interaction
# effect than an exhaustive all-pairs expansion over all 103 features,
# which would mostly multiply together pairs of near-noise features and
# give Lasso nothing but extra haystack to search.
_TOP_CORRELATED_FEATURES = [
    "ecg_qt_mean", "ecg_qtc_mean", "ecg_lf_hf_ratio", "ecg_sd1_sd2_ratio",
    "ecg_min", "ecg_max", "ecg_range", "ecg_std", "ecg_qt_std",
    "ecg_hr_mean", "breathing_std", "breathing_max",
]


def _pairwise_interaction_features(names, vals):
    """
    x_i * x_j for every pair among _TOP_CORRELATED_FEATURES (66 pairs from
    12 features), appended after the base features. Still fully compatible
    with the assignment's linear-model constraint: yhat = w0 + coef @ z
    stays linear in this final feature vector z, only the feature
    construction is nonlinear. Unlike a pure square (x_i^2 for all 103
    features -- tried and empirically discarded: Lasso selected zero of
    them on d3, since squaring an already-monotonic feature is nearly
    collinear with the original over this data's observed range and gives
    the model nothing new), a cross-feature product can encode a genuinely
    different signal -- e.g. how an ECG timing feature's relationship with
    glucose shifts under higher breathing variability -- that neither
    feature alone captures. NaNs propagate correctly on their own
    (nan * x == nan), so no extra guarding is needed here.
    """
    idx = {n: i for i, n in enumerate(names)}
    pair_names, pair_vals = [], []
    for a, b in itertools.combinations(_TOP_CORRELATED_FEATURES, 2):
        pair_names.append(f"{a}_x_{b}")
        pair_vals.append(vals[idx[a]] * vals[idx[b]])
    return pair_names, pair_vals


def make_features_for_window(record, feature_names_out):
    """
    record: dict-like with per-window 1D/2D arrays already sliced for ONE example,
            e.g. record["e4_bvp"] has shape (19200,), record["e4_acc"] has shape (9600, 3).
    feature_names_out: list to append feature names to (only populated on first call;
                        caller is responsible for only using this on the first row and
                        asserting consistency afterwards).
    Returns: 1D numpy array of feature values for this example.
    """
    names = []
    vals = []
    eps = 1e-6  # guards divide-by-zero in the motion-quality ratios below

    # --- BVP (E4 PPG) ---
    # Interpolated once and peak-detected once here; both are reused below
    # by _ptt_features instead of it re-running either step a second time.
    # bvp_stats is also kept for the bvp_motion_quality ratio further down.
    bvp = record["e4_bvp"]
    bvp_filled = _interpolate_nans(bvp)
    bvp_peaks = _bvp_peaks(bvp_filled, fs=64)
    bvp_stats = _stats(bvp, "bvp", names)
    vals += bvp_stats
    vals += _slope(bvp, "bvp", names)
    vals += _bvp_pulse_features(bvp_filled, bvp_peaks, fs=64, names=names)
    vals += _wavelet_energy_features(bvp_filled, 19200, "bvp", names)
    vals += _bvp_augmentation_index(bvp_filled, bvp_peaks, names)
    vals += _teager_energy_features(bvp_filled, "bvp", names)
    vals += _spectral_entropy_features(bvp_filled, fs=64, prefix="bvp", names=names)

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

    # --- Temperature ---
    # Uses detrended stats (see _stats_detrended), not raw _stats: resting
    # skin temperature is a fairly stable per-person baseline trait, so its
    # raw absolute level acts as a subject fingerprint much like the raw
    # ECG/breathing offsets did before detrending (confirmed empirically --
    # temp_mean/max/min topped the d3 cross-subject correlation table before
    # this fix). _slope/_block_trend_features are untouched: both are
    # already invariant to a constant per-subject offset (a shift cancels
    # out in a linear-fit slope and in a difference-of-block-means), so they
    # were never part of this leak.
    temp = record["e4_temp"]
    temp_filled = _interpolate_nans(temp)
    vals += _stats_detrended(temp_filled, fs=4, prefix="temp", names=names)
    vals += _slope(temp, "temp", names)
    vals += _block_trend_features(temp, "temp", names)

    # --- E4 accelerometer: squared magnitude a_sq(t) = ax^2+ay^2+az^2 ---
    # acc_sq_stats is kept for the bvp_motion_quality ratio just below.
    acc = record["e4_acc"]  # (T, 3)
    acc_sq = np.nansum(acc.astype(np.float64) ** 2, axis=1)
    acc_sq_stats = _stats(acc_sq, "e4_acc_sq", names)
    vals += acc_sq_stats

    # BVP signal-quality proxy: pulse-amplitude variability relative to
    # concurrent wrist motion energy. Motion artifact is the classic reason
    # PPG-derived features become unreliable, so this tells the model how
    # much to trust the bvp_* features in this specific window rather than
    # treating every window as equally clean.
    bvp_motion_quality = float(bvp_stats[1] / (acc_sq_stats[0] + eps))
    names.append("bvp_motion_quality")
    vals.append(bvp_motion_quality)

    # --- Zephyr ECG ---
    # Interpolated once here; reused by _stats_detrended and _ecg_r_peaks_nk
    # instead of each independently re-interpolating the same 75000-sample
    # signal. R-peaks are likewise detected once and shared with
    # _hrv_freq_features and _ptt_features. ecg_stats kept for
    # ecg_motion_quality further down.
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

    # ECG signal-quality proxy, same rationale as bvp_motion_quality above
    # but for the chest-strap ECG vs. chest motion.
    ecg_motion_quality = float(ecg_stats[1] / (zacc_sq_stats[0] + eps))
    names.append("ecg_motion_quality")
    vals.append(ecg_motion_quality)

    # --- Zephyr breathing ---
    # Interpolated once here; reused by _stats_detrended and
    # _breathing_rate_features instead of each re-interpolating separately.
    # block_trend/block_var use the raw (not detrended) signal -- both are
    # already invariant to a constant additive offset (see the temp comment
    # above), so this doesn't reopen the calibration-offset leak; it adds a
    # half-window amplitude/drift trend that _stats_detrended/resp_rate
    # don't otherwise capture.
    breathing = record["zephyr_breathing"]
    breathing_filled = _interpolate_nans(breathing)
    vals += _stats_detrended(breathing_filled, fs=25, prefix="breathing", names=names)
    breathing_rate_vals = _breathing_rate_features(breathing_filled, fs=25, names=names)
    vals += breathing_rate_vals
    vals += _block_trend_features(breathing, "breathing", names)

    # RSA power needs this window's own measured breathing rate (just
    # computed above) and the R-peaks from the ECG section above -- called
    # here, not in the ECG block, since resp_rate isn't known until now.
    resp_rate_bpm = breathing_rate_vals[0]
    vals += _hrv_rsa_power(ecg_r_peaks, fs_ecg=250, resp_rate_bpm=resp_rate_bpm, names=names)

    # --- Pairwise interaction terms (top-correlated features only) ---
    # Appended last so the base (linear) features keep the exact
    # names/order already relied on elsewhere (e.g. inspect_model, the
    # correlation printout).
    pair_names, pair_vals = _pairwise_interaction_features(names, vals)
    names += pair_names
    vals += pair_vals

    if not feature_names_out:
        feature_names_out.extend(names)
    else:
        assert feature_names_out == names, "Feature name/order mismatch between windows"

    return np.array(vals, dtype=np.float64)


def iter_participant_files(data_dir):
    return sorted(Path(data_dir).glob("*.npz"))


def build_feature_matrix(data_dir, has_target):
    """
    Processes one .npz file at a time to keep memory bounded.
    Returns (Z, y, feature_names) where y is None if has_target is False.
    """
    feature_names = []
    feature_rows = []
    targets = [] if has_target else None

    files = iter_participant_files(data_dir)
    if(DEBUG): print(f"[build_feature_matrix] {data_dir}: {len(files)} file(s) to process")

    for file_idx, path in enumerate(files, start=1):
        file_start = time.time()
        with np.load(path, allow_pickle=False) as data:
            n = data["e4_bvp"].shape[0]
            if(DEBUG): print(f"[build_feature_matrix] ({file_idx}/{len(files)}) {path.name}: {n} window(s)")

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

            progress_step = max(1, n // 5)  # ~5 progress lines per file
            for i in range(n):
                record = {k: v[i] for k, v in fields.items()}
                row = make_features_for_window(record, feature_names)
                feature_rows.append(row)
                if (i + 1) % progress_step == 0 or (i + 1) == n:
                    elapsed = time.time() - file_start
                    rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
                    if(DEBUG): print(f"    ... {i + 1}/{n} windows "
                          f"({elapsed:.1f}s elapsed, {rate:.1f} windows/s)")

            if has_target:
                targets.append(glucose)

        if(DEBUG): print(f"[build_feature_matrix] ({file_idx}/{len(files)}) {path.name} done "
              f"in {time.time() - file_start:.1f}s")

    Z = np.vstack(feature_rows)
    y = np.concatenate(targets) if has_target else None
    if(DEBUG): print(f"[build_feature_matrix] {data_dir}: all files done, Z shape={Z.shape}"
          + (f", y shape={y.shape}" if has_target else ""))
    return Z, y, feature_names

def cmd_train(protocol, train_dir, model_path, show_correlations=False, top_n=15):
    run_start = time.time()
    if(DEBUG): print(f"[cmd_train] protocol={protocol}: building training features from {train_dir}")
    t0 = time.time()
    Z, y, feature_names = build_feature_matrix(train_dir, has_target=True)
    if(DEBUG): print(f"[cmd_train] feature building done in {time.time() - t0:.1f}s")

    if show_correlations:
        # Reuses the Z/y already built above -- avoids a second, expensive
        # build_feature_matrix pass just to compute this diagnostic. Off by
        # default so the real graded CLI path (main() calling cmd_train with
        # 3 positional args) is completely unaffected.
        df = pd.DataFrame(Z, columns=feature_names)
        df["glucose"] = y
        corr = df.corr(numeric_only=True)["glucose"].drop("glucose")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        print(f"\n=== feature-glucose correlation (top {top_n}) ===")
        for name, r in corr.head(top_n).items():
            print(f"{name:25s} {r:8.4f}")
        print()

    # Z may contain NaN columns for windows where a modality had no valid
    # samples at all (see _stats/_slope). Impute with the per-feature median,
    # fit using training data only, per the assignment's leakage rules.
    # keep_empty_features=True: SimpleImputer's default (False) SILENTLY
    # DROPS a column entirely if it's NaN for every training window --
    # plausible now that fragile features (ptt_mean, block_trend, HRV, SCR)
    # correctly emit real NaN on failure. A dropped column would desync
    # Z_imputed's width from feature_names/coef without ever raising an
    # error -- only surfacing as a shape mismatch at actual grading time.
    if(DEBUG): print("[cmd_train] imputing missing values")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Z_imputed = imputer.fit_transform(Z)
    assert Z_imputed.shape[1] == len(feature_names), (
        f"Imputer changed feature count: {len(feature_names)} -> {Z_imputed.shape[1]}"
    )

    if(DEBUG): print("[cmd_train] scaling features")
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z_imputed)

    # LassoCV, not RidgeCV: diagnose_compare_models (see the dev-only
    # diagnose section) showed Lasso/ElasticNet meaningfully beating Ridge
    # specifically on d3 (cross-subject) -- ~7.5% relative NMAE gain over
    # baseline vs. Ridge's ~5.3%, using only 3 of 89 features (ecg_min,
    # ecg_lf_hf_ratio, breathing_std). d1 showed Ridge/Lasso essentially
    # tied, so there's no real downside to standardizing on Lasso here.
    # ElasticNet's CV-selected l1_ratio was ~0.99 (functionally pure L1)
    # with no measurable benefit over plain Lasso, so it's not worth the
    # extra hyperparameter/fit-time cost of searching l1_ratio too.
    if(DEBUG): print("[cmd_train] fitting LassoCV")
    t1 = time.time()
    model = LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, max_iter=50000, n_jobs=-1)
    with warnings.catch_warnings():
        # Non-convergence warnings for some alpha values in the CV grid --
        # the CV selection still picks the best-scoring alpha regardless.
        warnings.simplefilter("ignore")
        model.fit(Z_scaled, y)
    if(DEBUG): print(f"[cmd_train] LassoCV fit done in {time.time() - t1:.1f}s")

    # Convert standardized-space coefficients back to raw feature scale so that
    # saved (intercept, coef) act directly on the *unscaled* engineered features.
    # y_hat = intercept_s + coef_s . ((z - mean) / scale)
    #       = (intercept_s - sum(coef_s * mean / scale)) + sum((coef_s/scale) * z)
    coef_scaled = model.coef_
    coef_raw = coef_scaled / scaler.scale_
    intercept_raw = model.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)

    if show_correlations:
        # Report requirement (page 11, "Top five features"): importance must
        # be the |coef| *after standardizing features*, not raw |coef_raw| --
        # raw coefficients are scale-dependent (a feature with a tiny natural
        # std gets an inflated raw coefficient for the same standardized-space
        # effect, which is exactly what made eval_d.py's own raw-|coef| "top
        # features" printout misleading for d1: bvp_slope/bvp_wavelet_energy_l1
        # topped it purely from having near-zero natural scale, not real
        # importance). coef_scaled here *is* the correctly standardized
        # coefficient already computed above, before the raw-scale conversion
        # cmd_train must do for the saved pickle.
        order = np.argsort(-np.abs(coef_scaled))[:top_n]
        print(f"=== top {top_n} features by |standardized coef| (for report) ===")
        for i in order:
            print(f"{feature_names[i]:30s} std_coef={coef_scaled[i]:+10.4f}")
        print()

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

    if(DEBUG): print(f"[cmd_train] saving model to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(state, f)

    if(DEBUG): print(f"[train] protocol={protocol} n={Z.shape[0]} m={Z.shape[1]} "
          f"best_alpha={model.alpha_:.4g}")

    total = time.time() - run_start
    if(DEBUG): print(f"[cmd_train] TOTAL TIME: {total:.1f}s ({total / 60:.2f} min) -- "
          f"assignment limit is 30 min per protocol for train + feature_engineering combined")

    return y


def cmd_feature_engineering(protocol, test_dir, model_path, output_path):
    run_start = time.time()
    if(DEBUG): print(f"[cmd_feature_engineering] loading model from {model_path}")
    with open(model_path, "rb") as f:
        state = pickle.load(f)

    assert state["protocol"] == protocol, "Model/protocol mismatch"

    if(DEBUG): print(f"[cmd_feature_engineering] protocol={protocol}: building test features from {test_dir}")
    t0 = time.time()
    Z, _, feature_names = build_feature_matrix(test_dir, has_target=False)
    if(DEBUG): print(f"[cmd_feature_engineering] feature building done in {time.time() - t0:.1f}s")
    assert feature_names == state["feature_names"], "Feature name/order mismatch vs. training"

    if(DEBUG): print("[cmd_feature_engineering] imputing missing values")
    imputer = state["preprocessing_state"]["imputer"]
    Z_imputed = imputer.transform(Z)

    assert Z_imputed.shape[1] == len(state["feature_names"]), (
        f"Imputer output width {Z_imputed.shape[1]} != "
        f"len(feature_names) {len(state['feature_names'])}"
    )
    assert np.isfinite(Z_imputed).all(), "Non-finite values remain in final feature matrix"

    if(DEBUG): print(f"[cmd_feature_engineering] saving features to {output_path}")
    np.save(output_path, Z_imputed)
    if(DEBUG): print(f"[feature_engineering] protocol={protocol} n={Z_imputed.shape[0]} m={Z_imputed.shape[1]}")

    total = time.time() - run_start
    if(DEBUG): print(f"[cmd_feature_engineering] TOTAL TIME: {total:.1f}s ({total / 60:.2f} min) -- "
          f"assignment limit is 30 min per protocol for train + feature_engineering combined")

def nmae(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))


def nmse(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_mean) ** 2)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def inspect_model(model_path):
    """
    Dev-only diagnostic: prints every saved feature's training-set median,
    pulled straight from the pickled SimpleImputer -- a cheap sanity check
    on feature plausibility (e.g. is ecg_hr_mean a real heart rate?) without
    needing to reload/reprocess any raw data. Not part of the graded CLI.
    """
    with open(model_path, "rb") as f:
        state = pickle.load(f)

    imputer = state["preprocessing_state"]["imputer"]
    print(f"protocol={state['protocol']}  n_features={len(state['feature_names'])}")
    for name, median in zip(state["feature_names"], imputer.statistics_):
        print(f"{name:25s} median={median:12.4f}")

    return state


def diagnose_eval(protocol, train_dir, test_dir, model_path):
    """
    Dev-only: trains (with feature-glucose correlation printing), then
    evaluates against the *dev* test folder's glucose field -- present in
    local development copies of the data, absent from the real hidden test
    set -- to report NMAE/NMSE for the model vs. a median-training-target
    baseline. This is exactly the comparison the report's "Median baseline
    comparison" section asks for. Not part of the graded CLI surface
    (train/feature_engineering) -- do not point this at the real test set.
    """
    y_train = cmd_train(protocol, train_dir, model_path, show_correlations=True)

    with open(model_path, "rb") as f:
        state = pickle.load(f)

    median_pred = np.median(y_train)

    if(DEBUG): print(f"[diagnose eval] building test features (dev-only, expects glucose present) from {test_dir}")
    Z_test, y_test, feature_names = build_feature_matrix(test_dir, has_target=True)
    assert feature_names == state["feature_names"], "Feature name/order mismatch vs. training"

    imputer = state["preprocessing_state"]["imputer"]
    Z_test_imputed = imputer.transform(Z_test)
    assert np.isfinite(Z_test_imputed).all(), "Non-finite values in test feature matrix"

    y_pred = state["intercept"] + Z_test_imputed @ state["coef"]
    y_pred_median = np.full_like(y_test, median_pred)

    print(f"\n=== Protocol {protocol} ===")
    print(f"n_train={len(y_train)}  n_test={len(y_test)}  n_features={len(feature_names)}")
    print(f"median(y_train) = {median_pred:.2f} mg/dL")
    print(f"mean(y_test) = {np.mean(y_test):.2f} mg/dL   std(y_test) = {np.std(y_test):.2f} mg/dL")
    print()
    print(f"{'':12s} {'NMAE':>10s} {'NMSE':>10s} {'MAE(mg/dL)':>12s} {'RMSE(mg/dL)':>12s}")
    print(f"{'model':12s} {nmae(y_test, y_pred):10.4f} {nmse(y_test, y_pred):10.4f} "
          f"{mae(y_test, y_pred):12.2f} {rmse(y_test, y_pred):12.2f}")
    print(f"{'median':12s} {nmae(y_test, y_pred_median):10.4f} {nmse(y_test, y_pred_median):10.4f} "
          f"{mae(y_test, y_pred_median):12.2f} {rmse(y_test, y_pred_median):12.2f}")

    return state, y_test, y_pred


def diagnose_compare_models(protocol, train_dir, test_dir):
    """
    Dev-only: builds train/test features exactly once, then fits Ridge,
    Lasso, and ElasticNet (all explicitly permitted final models per the
    assignment) on the identical preprocessed data and reports NMAE/NMSE
    for each on the dev test set, plus the median baseline. Model fitting
    itself is cheap relative to feature extraction (RidgeCV fits in
    ~0.1s), so comparing three models costs almost nothing once Z is
    already built -- this does NOT save a model file, it's purely a
    comparison. Uses the dev test dir's glucose field (present locally,
    absent from the real hidden test set) -- do not point this at the
    real test set.

    Fits directly in standardized (scaled) feature space and predicts
    there too (model.predict(Z_test_scaled)), rather than converting
    coefficients back to raw-feature space like cmd_train does for the
    saved pickle -- mathematically identical predictions, just simpler
    since nothing here gets serialized.
    """
    if(DEBUG): print(f"[diagnose compare] protocol={protocol}: building training features from {train_dir}")
    t0 = time.time()
    Z, y, feature_names = build_feature_matrix(train_dir, has_target=True)
    if(DEBUG): print(f"[diagnose compare] feature building done in {time.time() - t0:.1f}s")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Z_imputed = imputer.fit_transform(Z)
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z_imputed)

    if(DEBUG): print(f"[diagnose compare] building test features from {test_dir}")
    t0 = time.time()
    Z_test, y_test, test_feature_names = build_feature_matrix(test_dir, has_target=True)
    if(DEBUG): print(f"[diagnose compare] feature building done in {time.time() - t0:.1f}s")
    assert test_feature_names == feature_names, "Feature name/order mismatch vs. training"

    Z_test_imputed = imputer.transform(Z_test)
    Z_test_scaled = scaler.transform(Z_test_imputed)

    median_pred = np.median(y)
    y_pred_median = np.full_like(y_test, median_pred)

    models = {
        "ridge": RidgeCV(alphas=np.logspace(-3, 6, 19)),
        "lasso": LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, max_iter=50000, n_jobs=-1),
        "elasticnet": ElasticNetCV(alphas=np.logspace(-4, 1, 30),
                                    l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1.0],
                                    cv=5, max_iter=50000, n_jobs=-1),
        # Robust regression: downweights outlier residuals (Huber loss)
        # instead of squared-error's full quadratic penalty. Worth trying
        # since real physiological windows likely have occasional
        # artifacts/noise even after the NaN-safety handling elsewhere --
        # if a handful of bad windows are pulling Ridge/Lasso's squared-loss
        # fit off, Huber should show a visible improvement here. Still
        # produces intercept + coef, so it's a permitted final model.
        "huber": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=5000),
    }

    print(f"\n=== Protocol {protocol}: model comparison (n_train={len(y)}  "
          f"n_test={len(y_test)}  n_features={len(feature_names)}) ===")
    print(f"{'model':12s} {'NMAE':>10s} {'NMSE':>10s} {'n_nonzero':>10s} {'fit_time':>10s}  details")
    print(f"{'median':12s} {nmae(y_test, y_pred_median):10.4f} {nmse(y_test, y_pred_median):10.4f} "
          f"{'--':>10s} {'--':>10s}")

    results = {}
    for name, model in models.items():
        t_fit = time.time()
        with warnings.catch_warnings():
            # Lasso/ElasticNet can warn about non-convergence for some
            # alpha/l1_ratio combinations in the CV grid; the CV selection
            # itself still picks the best-scoring combination regardless,
            # so this is noise for a diagnostic comparison, not a real
            # problem worth surfacing here.
            warnings.simplefilter("ignore")
            model.fit(Z_scaled, y)
        fit_time = time.time() - t_fit

        y_pred = model.predict(Z_test_scaled)
        n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-10))

        details = []
        if hasattr(model, "alpha_"):
            details.append(f"alpha={model.alpha_:.4g}")
        if hasattr(model, "l1_ratio_"):
            details.append(f"l1_ratio={model.l1_ratio_:.3g}")

        print(f"{name:12s} {nmae(y_test, y_pred):10.4f} {nmse(y_test, y_pred):10.4f} "
              f"{n_nonzero:10d} {fit_time:9.1f}s  {' '.join(details)}")

        # For sparse models, print which features got zeroed out entirely --
        # direct, data-driven evidence for which engineered features are
        # redundant/uninformative vs. worth keeping (e.g. useful for
        # deciding whether the wavelet-energy features are pulling weight).
        if n_nonzero < len(feature_names):
            dropped = [f for f, c in zip(feature_names, model.coef_) if abs(c) <= 1e-10]
            print(f"             dropped by {name} ({len(dropped)}): {', '.join(dropped)}")

        results[name] = model

    # Relaxed Lasso: Lasso's L1 penalty shrinks the coefficients of the
    # features it *keeps*, not just zeroes out the ones it drops -- that
    # shrinkage is a real accuracy cost, not just a sparsity mechanism.
    # Refitting an unregularized OLS using only Lasso's selected support
    # removes that bias. Well-established technique (a.k.a. post-lasso
    # OLS / relaxed lasso) for recovering accuracy Lasso's penalty gives up.
    lasso_model = results.get("lasso")
    if lasso_model is not None:
        support = np.abs(lasso_model.coef_) > 1e-10
        n_support = int(np.sum(support))
        if 0 < n_support < Z_scaled.shape[0]:
            t_fit = time.time()
            refit = LinearRegression()
            refit.fit(Z_scaled[:, support], y)
            fit_time = time.time() - t_fit
            y_pred = refit.predict(Z_test_scaled[:, support])
            support_names = [f for f, s in zip(feature_names, support) if s]
            print(f"{'relaxed-lasso':12s} {nmae(y_test, y_pred):10.4f} {nmse(y_test, y_pred):10.4f} "
                  f"{n_support:10d} {fit_time:9.1f}s  OLS refit on: {', '.join(support_names)}")
            results["relaxed-lasso"] = refit

    return results


def main():
    usage = (
        "usage:\n"
        "  part_d.py train <protocol> <train_dir> <model_path>\n"
        "  part_d.py feature_engineering <protocol> <test_dir> <model_path> <output_path>\n"
        "\n"
        "  dev-only diagnostics (not part of the graded interface):\n"
        "  part_d.py diagnose train <protocol> <train_dir> <model_path>\n"
        "  part_d.py diagnose inspect <model_path>\n"
        "  part_d.py diagnose eval <protocol> <train_dir> <test_dir> <model_path>\n"
        "  part_d.py diagnose compare <protocol> <train_dir> <test_dir>"
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
    elif mode == "diagnose":
        if len(sys.argv) < 3:
            print(usage)
            sys.exit(1)
        sub = sys.argv[2]
        if sub == "train":
            _, _, _, protocol, train_dir, model_path = sys.argv
            cmd_train(protocol, train_dir, model_path, show_correlations=True)
        elif sub == "inspect":
            _, _, _, model_path = sys.argv
            inspect_model(model_path)
        elif sub == "eval":
            _, _, _, protocol, train_dir, test_dir, model_path = sys.argv
            diagnose_eval(protocol, train_dir, test_dir, model_path)
        elif sub == "compare":
            _, _, _, protocol, train_dir, test_dir = sys.argv
            diagnose_compare_models(protocol, train_dir, test_dir)
        else:
            print(usage)
            sys.exit(1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
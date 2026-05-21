import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats
import neurokit2 as nk

from .signal_quality import flatline_ratio

Z_SCORE_THRESHOLD = 10.0
SNR_WINDOW_SEC    = 0.1


def get_ecg_features(ecg, time_in_sec, fs):
    """Extract HR statistics, HRV, SNR, and flatline ratio from a raw ECG segment.

    Flatline is checked first — if detected, SNR is set to 0 and cardiac metrics
    are returned as NaN without attempting R-peak detection.

    Returns
    -------
    np.ndarray of shape (6,):
        [hr_mean, hr_max, hr_min, hrv, snr, flatline_ratio]
    """
    # --- Flatline check: short-circuit with SNR=0 if signal is flat ---
    flat = flatline_ratio(ecg)
    if flat == 1.0:
        return np.array([np.nan, np.nan, np.nan, np.nan, 0.0, flat])

    b, a = butter(4, (0.25, 30), 'bandpass', fs=fs)
    ecg_filt    = filtfilt(b, a, ecg, axis=0)
    ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)

    # --- R-peak dependent cardiac metrics ---
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="engzeemod2012")
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat])

    rr_times = time_in_sec[rpeaks['ECG_R_Peaks']]
    if len(rr_times) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat])

    # Heart rate from RR intervals
    d_rr = np.diff(rr_times)
    heart_rate = 60 / d_rr
    if heart_rate.size == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat])

    valid_hr   = heart_rate[~np.isnan(heart_rate)]
    heart_rate = valid_hr[np.abs(stats.zscore(valid_hr)) <= Z_SCORE_THRESHOLD]

    hr_mean = np.nanmean(heart_rate)
    hr_min  = np.nanmin(heart_rate)
    hr_max  = np.nanmax(heart_rate)

    # HRV: RMSSD of successive RR differences
    d_rr_ms  = 1000 * d_rr
    d_d_rr   = np.diff(d_rr_ms)
    valid_dd = d_d_rr[~np.isnan(d_d_rr)]
    valid_dd = valid_dd[np.abs(stats.zscore(valid_dd)) <= Z_SCORE_THRESHOLD]
    hrv      = np.sqrt(np.nanmean(np.square(valid_dd)))

    # SNR via ±SNR_WINDOW_SEC window around each R-peak
    raw_segs, clean_segs = [], []
    for t_r in rr_times:
        idx = np.where(
            (time_in_sec >= t_r - SNR_WINDOW_SEC) &
            (time_in_sec <= t_r + SNR_WINDOW_SEC)
        )[0]
        idx = idx[(idx >= 0) & (idx < len(ecg))]
        if len(idx) > 0:
            raw_segs.extend(ecg[idx])
            clean_segs.extend(ecg_cleaned[idx])

    raw_arr   = np.array(raw_segs)
    clean_arr = np.array(clean_segs)
    sig_power  = np.var(raw_arr)
    noise_power = np.var(raw_arr - clean_arr)
    snr = 10 * np.log10(sig_power / noise_power)

    return np.array([hr_mean, hr_max, hr_min, hrv, snr, flat])

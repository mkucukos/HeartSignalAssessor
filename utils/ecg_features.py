import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats
import neurokit2 as nk

Z_SCORE_THRESHOLD = 5.0
SNR_WINDOW_SEC = 0.1


def get_ecg_features(ecg, time_in_sec, fs):
    """Extract HR statistics, HRV, and SNR from a raw ECG segment."""
    try:
        b, a = butter(4, (0.25, 30), 'bandpass', fs=fs)
        ecg_filt = filtfilt(b, a, ecg, axis=0)
        ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="engzeemod2012")
    except Exception as exc:
        raise ValueError("Error processing ECG signal: " + str(exc))

    rr_times = time_in_sec[rpeaks['ECG_R_Peaks']]
    if len(rr_times) == 0:
        raise ValueError("No R-peaks detected in ECG signal.")

    # Heart rate from RR intervals
    d_rr = np.diff(rr_times)
    heart_rate = 60 / d_rr
    if heart_rate.size == 0:
        raise ValueError("Error computing heart rate from ECG signal.")

    valid_hr = heart_rate[~np.isnan(heart_rate)]
    heart_rate = valid_hr[np.abs(stats.zscore(valid_hr)) <= Z_SCORE_THRESHOLD]

    hr_mean = np.nanmean(heart_rate)
    hr_min = np.nanmin(heart_rate)
    hr_max = np.nanmax(heart_rate)

    # HRV: RMSSD of successive RR differences
    d_rr_ms = 1000 * d_rr
    d_d_rr_ms = np.diff(d_rr_ms)
    valid_dd = d_d_rr_ms[~np.isnan(d_d_rr_ms)]
    valid_dd = valid_dd[np.abs(stats.zscore(valid_dd)) <= Z_SCORE_THRESHOLD]
    heart_rate_variability = np.sqrt(np.nanmean(np.square(valid_dd)))

    # SNR via ±SNR_WINDOW_SEC window around each R-peak
    raw_segments, clean_segments = [], []
    for t_r in rr_times:
        indices = np.where(
            (time_in_sec >= t_r - SNR_WINDOW_SEC) &
            (time_in_sec <= t_r + SNR_WINDOW_SEC)
        )[0]
        indices = indices[(indices >= 0) & (indices < len(ecg))]
        if len(indices) > 0:
            raw_segments.extend(ecg[indices])
            clean_segments.extend(ecg_cleaned[indices])

    raw_arr = np.array(raw_segments)
    clean_arr = np.array(clean_segments)
    signal_power = np.var(raw_arr)
    noise_power = np.var(raw_arr - clean_arr)
    snr_values = 10 * np.log10(signal_power / noise_power)

    return np.array([hr_mean, hr_max, hr_min, heart_rate_variability, snr_values])

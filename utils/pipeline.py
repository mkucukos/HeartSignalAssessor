import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt

from .ecg_features import get_ecg_features
from .noise import get_noise_std, is_flatline_period


def generate_ecg_data(
    fs: int = 250,
    num_frames: int = 750,
    window_size: int = 30,
    plot_tail: int = 10,
    duration_per_frame: int = 15,
) -> pd.DataFrame:
    """Simulate ECG frames and extract signal features.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    num_frames : int
        Number of simulation frames to generate.
    window_size : int
        Feature extraction window length in seconds.
    plot_tail : int
        Seconds of ECG to keep in the animation plot window.
    duration_per_frame : int
        Duration of each simulated ECG segment in seconds.

    Returns
    -------
    df : pd.DataFrame
        One row per frame containing raw signal slices, features, and metadata.
    """
    print("Generating cumulative ECG data...")

    all_rows: list[dict] = []
    cumulative_ecg: list[float] = []
    cumulative_time: list[float] = []

    for frame_idx in range(num_frames):
        print(f"Processing frame {frame_idx}...")

        # --- Physiologically realistic HR variation ---
        breathing_effect = 3.0 * np.sin(2 * np.pi * (0.15 * frame_idx) / 60)
        time_trend = 8.0 * np.sin(2 * np.pi * frame_idx / (num_frames * 0.6))
        natural_variation = np.random.normal(0, 3)
        random_walk = np.random.normal(0, 1) if frame_idx > 0 else 0.0
        current_hr = float(np.clip(
            70 + natural_variation + breathing_effect + time_trend + random_walk,
            55, 95,
        ))

        n_samples = duration_per_frame * fs

        if is_flatline_period(frame_idx):
            # Flatline: constant zero signal with sub-threshold noise (triggers flatline_ratio = 1.0)
            ecg_noisy = np.random.normal(0, 1e-8, n_samples)
            actual_noise_std = 0.0
        else:
            np.random.seed(frame_idx * 42 + 789)
            ecg_segment = nk.ecg_simulate(
                duration_per_frame, sampling_rate=fs, heart_rate=current_hr
            )
            np.random.seed(None)

            # --- Multi-component progressive noise ---
            base_noise_std = get_noise_std(frame_idx)
            actual_noise_std = base_noise_std * np.random.uniform(0.8, 1.2)

            gaussian_noise = np.random.normal(0, actual_noise_std, len(ecg_segment))

            if actual_noise_std > 0.1:
                hf_factor = min(0.3, (actual_noise_std - 0.1) * 0.5)
                hf_raw = hf_factor * np.random.normal(0, 1, len(ecg_segment))
                b_hf, a_hf = butter(4, 30, "highpass", fs=fs)
                hf_noise: np.ndarray | float = filtfilt(b_hf, a_hf, hf_raw)
            else:
                hf_noise = 0.0

            if actual_noise_std > 0.2:
                lf_factor = min(0.2, (actual_noise_std - 0.2) * 0.3)
                lf_drift: np.ndarray | float = lf_factor * np.sin(
                    2 * np.pi * 0.5 * np.arange(len(ecg_segment)) / fs
                )
            else:
                lf_drift = 0.0

            ecg_noisy = ecg_segment + gaussian_noise + hf_noise + lf_drift

        # --- Append to cumulative signal ---
        t_start = len(cumulative_ecg) / fs
        cumulative_ecg.extend(ecg_noisy.tolist())
        seg_time = np.arange(t_start, t_start + duration_per_frame, 1 / fs)[: len(ecg_noisy)]
        cumulative_time.extend(seg_time.tolist())

        # --- Feature extraction over 30-second sliding window ---
        min_samples = window_size * fs
        features_valid = False
        features = np.full(6, np.nan)

        if len(cumulative_ecg) >= min_samples:
            ecg_win = np.array(cumulative_ecg[-min_samples:])
            t_win   = np.array(cumulative_time[-min_samples:])
            t_win   = t_win - t_win[0]
            features = get_ecg_features(ecg_win, t_win, fs)
            features_valid = bool(np.all(np.isfinite(features[:5])))

        # --- 10-second plot tail (zero-referenced time axis) ---
        tail_samples = plot_tail * fs
        if len(cumulative_ecg) >= tail_samples:
            ecg_plot = cumulative_ecg[-tail_samples:]
            time_plot = cumulative_time[-tail_samples:]
        else:
            ecg_plot = cumulative_ecg.copy()
            time_plot = cumulative_time.copy()

        time_plot = [t - time_plot[0] for t in time_plot] if time_plot else []

        all_rows.append({
            "frame":               frame_idx,
            "time_global":         len(cumulative_ecg) / fs,
            "noise_std":           actual_noise_std,
            "base_noise_std":      base_noise_std,
            "current_hr":          current_hr,
            "ecg_plot":            ecg_plot,
            "time_plot":           time_plot,
            "buffer_full":         len(cumulative_ecg) >= min_samples,
            "cumulative_duration": len(cumulative_ecg) / fs,
            "segment_duration":    duration_per_frame,
            "hr_mean":             features[0],
            "hr_max":              features[1],
            "hr_min":              features[2],
            "hrv":                 features[3],
            "snr":                 features[4],
            "flatline_ratio":      features[5],
            "is_flatline_period":  is_flatline_period(frame_idx),
            "features_valid":      features_valid,
        })

    return pd.DataFrame(all_rows)

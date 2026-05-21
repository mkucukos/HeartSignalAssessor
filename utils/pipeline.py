import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from .ecg_features import get_ecg_features
from .noise import get_noise_std


def generate_ecg_data(
    model,
    fs: int = 250,
    num_frames: int = 200,
    window_size: int = 30,
    plot_tail: int = 10,
    duration_per_frame: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate ECG frames, extract features, and compute rolling ML predictions.

    Parameters
    ----------
    model : callable
        Pre-loaded TensorFlow SavedModel.
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
    valid_features : pd.DataFrame
        Subset of df where features_valid is True, with an added 'prediction' column.
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
        features = np.full(5, np.nan)

        if len(cumulative_ecg) >= min_samples:
            ecg_win = np.array(cumulative_ecg[-min_samples:])
            t_win = np.array(cumulative_time[-min_samples:])
            t_win = t_win - t_win[0]
            try:
                features = get_ecg_features(ecg_win, t_win, fs)
                features_valid = bool(np.all(np.isfinite(features)))
            except Exception:
                pass

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
            "features_valid":      features_valid,
        })

    df = pd.DataFrame(all_rows)

    # --- Rolling-normalized ML predictions on valid frames only ---
    valid_features = df[df["features_valid"]].copy()

    if len(valid_features) > 0:
        hr_means = valid_features["hr_mean"].tolist()
        hr_maxs  = valid_features["hr_max"].tolist()
        hr_mins  = valid_features["hr_min"].tolist()
        hrvs     = valid_features["hrv"].tolist()

        predictions: list[float] = []
        for i in range(len(hr_means)):
            data = np.column_stack([
                hr_means[: i + 1], hr_maxs[: i + 1],
                hr_mins[: i + 1], hrvs[: i + 1],
            ])
            feat_vec = StandardScaler().fit_transform(data)[-1].reshape(1, -1)
            raw_pred = model(feat_vec)

            if hasattr(raw_pred, "numpy"):
                pred_val = float(raw_pred.numpy()[0][0])
            elif isinstance(raw_pred, dict):
                pred_val = float(next(iter(raw_pred.values()))[0][0])
            else:
                pred_val = float(raw_pred[0][0])

            predictions.append(pred_val)

        valid_features = valid_features.copy()
        valid_features["prediction"] = predictions

    return df, valid_features

from __future__ import annotations

import numpy as np


def flatline_ratio(signal: np.ndarray, eps: float = 1e-6) -> float:
    """Detect whether a signal epoch is flatlined.

    Sourced from physio-qc-toolkit (github.com/mkucukos/physio-qc-toolkit).

    Returns 1.0 if the epoch is flatline-like, 0.0 otherwise.
    Three independent conditions are checked — any one is sufficient:
      - Variance < 1e-12  (near-zero power)
      - Peak-to-peak < 1e-6  (no measurable amplitude)
      - >98% of consecutive sample differences < eps  (stuck ADC / repeated values)

    Parameters
    ----------
    signal : array-like
        1-D signal epoch.
    eps : float
        Difference threshold for the repeat-ratio check.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < 2:
        return 1.0

    if (
        np.var(sig) < 1e-12
        or np.ptp(sig) < 1e-6
        or np.mean(np.abs(np.diff(sig)) < eps) > 0.98
    ):
        return 1.0
    return 0.0


def baseline_wander_ratio(signal: np.ndarray, fs: int) -> float:
    """Fraction of signal power below 0.3 Hz (baseline wander band).

    Sourced from physio-qc-toolkit (github.com/mkucukos/physio-qc-toolkit).

    A high ratio indicates dominant low-frequency drift relative to the
    cardiac band. Typical threshold: ratio > 0.30 signals problematic wander.

    Parameters
    ----------
    signal : array-like
        1-D signal epoch.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    float
        Power ratio in [0, 1], or np.nan if the epoch is too short.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < fs:
        return np.nan

    freqs = np.fft.rfftfreq(sig.size, 1 / fs)
    psd   = np.abs(np.fft.rfft(sig)) ** 2
    total = psd.sum()
    if total == 0:
        return np.nan
    return float(psd[freqs <= 0.30].sum() / total)

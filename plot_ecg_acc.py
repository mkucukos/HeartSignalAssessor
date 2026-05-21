"""
ECG & ACC Processor — Vivalink VV330 chest patch

Plots produced
──────────────
01_ecg_snr.png          Full ECG (denoised) + windowed SNR
02_overview.png         ECG amplitude · HR · RMSSD · SNR
03_acc.png              ACC X / Y / Z / magnitude
04_ecg_acc_synopsis.png ECG raw + denoised + ACC (first 30 s)
05_snr_examples.png     5 raw-ECG windows spanning SNR range
"""

import ast
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy import stats

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent
RAW_PATH     = DATA_DIR / "ECG Sample_Raw.csv"
DEN_PATH     = DATA_DIR / "ECG Sample_Denoised.csv"
OUT_DIR      = DATA_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

ECG_FS       = 128     # Hz
ACC_FS       = 5       # Hz
SNR_WINDOW_S = 30.0    # seconds
SNR_STEP_S   = 5.0     # seconds

# bandpass once, reused everywhere
_B, _A = butter(4, [0.25, 30.0], btype="bandpass", fs=ECG_FS)

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         9,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _parse(s):
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return []


def load(path: Path) -> dict:
    df  = pd.read_csv(path)
    mag = float(df["Magnification"].iloc[0])

    # CSV timestamps have only 2 unique values across 3229 rows (precision lost).
    # Each row = exactly 128 ECG samples @ 128 Hz = 1 s → reconstruct from index.
    t_pkt = np.arange(len(df), dtype=float)   # seconds

    # ── ECG ───────────────────────────────────────────────────────────────────
    ecg_v, t_ecg = [], []
    for i, cell in enumerate(df["ECG"]):
        s = _parse(cell)
        if not s:
            continue
        t_ecg.extend(t_pkt[i] + np.arange(len(s)) / ECG_FS)
        ecg_v.extend(s)
    t_ecg = np.array(t_ecg)
    ecg   = np.array(ecg_v) / mag

    # ── ACC ───────────────────────────────────────────────────────────────────
    ax, ay, az, t_acc = [], [], [], []
    for i, cell in enumerate(df["ACC"]):
        for j, s in enumerate(_parse(cell)):
            if not isinstance(s, dict):
                continue
            t_acc.append(t_pkt[i] + j / ACC_FS)
            ax.append(s.get("x", np.nan))
            ay.append(s.get("y", np.nan))
            az.append(s.get("z", np.nan))
    t_acc = np.array(t_acc)
    acc_x = np.array(ax);  acc_y = np.array(ay);  acc_z = np.array(az)
    acc_m = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # ── HR & RMSSD (packet level) ─────────────────────────────────────────────
    hr = df["HR"].values.astype(float)
    hr[(hr < 20) | (hr > 300)] = np.nan

    rmssd = df["RMSSD"].values.astype(float)
    rmssd[rmssd <= 0] = np.nan

    # ── RRI (beat level) ──────────────────────────────────────────────────────
    rri_t, rri_v = [], []
    for i, cell in enumerate(df["RRI"]):
        vals = [v for v in _parse(cell) if v > 0]
        for j, v in enumerate(vals):
            rri_t.append(t_pkt[i] + j / max(len(vals), 1))
            rri_v.append(v)

    return dict(
        t_ecg=t_ecg, ecg=ecg,
        t_acc=t_acc, acc_x=acc_x, acc_y=acc_y, acc_z=acc_z, acc_m=acc_m,
        t_hr=t_pkt,  hr=hr,
        t_rmssd=t_pkt, rmssd=rmssd,
        t_rri=np.array(rri_t), rri=np.array(rri_v),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ECG FEATURES  (HeartSignalAssessor — ecg_features.py + signal_quality.py)
# ══════════════════════════════════════════════════════════════════════════════

Z_SCORE_THRESHOLD = 10.0
SNR_WINDOW_SEC    = 0.1


def _flatline_ratio(signal: np.ndarray, eps: float = 1e-6) -> float:
    """Inlined from HeartSignalAssessor/utils/signal_quality.py."""
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


def get_ecg_features(ecg: np.ndarray, time_in_sec: np.ndarray, fs: int) -> np.ndarray:
    """
    Adapted from HeartSignalAssessor/utils/ecg_features.py.
    Added: inversion metric via nk.ecg_invert (7th element).

    Returns
    -------
    np.ndarray of shape (7,):
        [hr_mean, hr_max, hr_min, hrv, snr, flatline_ratio, inverted]
    """
    flat = _flatline_ratio(ecg)
    if flat == 1.0:
        return np.array([np.nan, np.nan, np.nan, np.nan, 0.0, flat, np.nan])

    # --- inversion check (nk.ecg_invert) ---
    _, inverted = nk.ecg_invert(ecg, sampling_rate=fs)
    inv_flag = float(inverted)   # 1.0 = inverted, 0.0 = normal

    if inverted:
        # skip cardiac metrics for inverted windows; SNR = NaN
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat, inv_flag])

    b, a        = butter(4, (0.25, 30), "bandpass", fs=fs)
    ecg_filt    = filtfilt(b, a, ecg, axis=0)
    ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)

    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs,
                                  method="engzeemod2012")
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat, inv_flag])

    rr_times = time_in_sec[rpeaks["ECG_R_Peaks"]]
    if len(rr_times) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat, inv_flag])

    d_rr       = np.diff(rr_times)
    heart_rate = 60 / d_rr
    if heart_rate.size == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, flat, inv_flag])

    valid_hr   = heart_rate[~np.isnan(heart_rate)]
    heart_rate = valid_hr[np.abs(stats.zscore(valid_hr)) <= Z_SCORE_THRESHOLD]

    hr_mean = np.nanmean(heart_rate)
    hr_min  = np.nanmin(heart_rate)
    hr_max  = np.nanmax(heart_rate)

    d_rr_ms  = 1000 * d_rr
    d_d_rr   = np.diff(d_rr_ms)
    valid_dd = d_d_rr[~np.isnan(d_d_rr)]
    if valid_dd.size > 1:
        valid_dd = valid_dd[np.abs(stats.zscore(valid_dd)) <= Z_SCORE_THRESHOLD]
    hrv = np.sqrt(np.nanmean(np.square(valid_dd))) if valid_dd.size else np.nan

    raw_segs, clean_segs = [], []
    for t_r in rr_times:
        idx = np.where(
            (time_in_sec >= t_r - SNR_WINDOW_SEC) &
            (time_in_sec <= t_r + SNR_WINDOW_SEC)
        )[0]
        idx = idx[(idx >= 0) & (idx < len(ecg))]
        if len(idx):
            raw_segs.extend(ecg[idx])
            clean_segs.extend(ecg_cleaned[idx])

    raw_arr     = np.array(raw_segs)
    sig_power   = np.var(raw_arr)
    noise_power = np.var(raw_arr - np.array(clean_segs))
    snr = (10 * np.log10(sig_power / noise_power)
           if sig_power > 1e-12 and noise_power > 1e-12 else np.nan)

    return np.array([hr_mean, hr_max, hr_min, hrv, snr, flat, inv_flag])


def compute_windows(ecg: np.ndarray,
                    window_s: float = SNR_WINDOW_S,
                    step_s:   float = SNR_STEP_S):
    """
    Slide a window over the full ECG and call get_ecg_features per window.

    Returns
    -------
    t_centres : (N,)  seconds at window centre
    features  : (N,7) columns = [hr_mean, hr_max, hr_min, hrv, snr, flat, inv]
    starts    : (N,)  sample index of each window start
    """
    win  = int(window_s * ECG_FS)
    step = int(step_s   * ECG_FS)
    t_in_win = np.arange(win) / ECG_FS

    t_centres, feat_rows, starts = [], [], []
    for s in range(0, len(ecg) - win + 1, step):
        seg = ecg[s : s + win]
        t_centres.append((s + win / 2) / ECG_FS)
        starts.append(s)
        try:
            feat_rows.append(get_ecg_features(seg, t_in_win, ECG_FS))
        except Exception:
            feat_rows.append(np.full(7, np.nan))

    return (np.array(t_centres),
            np.array(feat_rows),
            np.array(starts))


# ── save helper ────────────────────────────────────────────────────────────────
def save(fig, name: str):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 01 — Full ECG (denoised) + windowed SNR
# ══════════════════════════════════════════════════════════════════════════════

def plot_ecg_snr(den, t_snr, snr, inverted):
    ds  = 4
    t_e = den["t_ecg"][::ds] / 60
    ecg = den["ecg"][::ds]
    t_s = t_snr / 60
    v   = ~np.isnan(snr)
    n_inv = inverted.sum()

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
    )
    fig.suptitle("Full ECG Recording + Windowed SNR", fontweight="bold", fontsize=12)

    axes[0].plot(t_e, ecg, lw=0.4, color="#2980b9", label="ECG (denoised)")
    axes[0].set_ylim(-2, 2)
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].tick_params(labelbottom=False)

    # shade inverted regions
    for i, (t_c, inv) in enumerate(zip(t_s, inverted)):
        if inv:
            axes[0].axvspan(t_c - SNR_STEP_S / 60 / 2,
                            t_c + SNR_STEP_S / 60 / 2,
                            color="#9b59b6", alpha=0.25, lw=0)
            axes[1].axvspan(t_c - SNR_STEP_S / 60 / 2,
                            t_c + SNR_STEP_S / 60 / 2,
                            color="#9b59b6", alpha=0.25, lw=0)

    axes[1].plot(t_s[v], snr[v], lw=1.1, color="#e67e22", label="SNR")
    axes[1].fill_between(t_s[v], snr[v], snr[v].min(), alpha=0.18, color="#e67e22")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_title(
        f"30 s window · 5 s step  ·  "
        f"median {np.nanmedian(snr):.1f} dB · "
        f"min {np.nanmin(snr):.1f} dB · "
        f"max {np.nanmax(snr):.1f} dB  ·  "
        f"{n_inv} inverted windows (purple)",
        fontsize=8, loc="left", pad=2,
    )
    inv_patch = mpatches.Patch(color="#9b59b6", alpha=0.5, label=f"Inverted ({n_inv})")
    axes[1].legend(handles=[
        plt.Line2D([0], [0], color="#e67e22", lw=1.5, label="SNR"),
        inv_patch,
    ], loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    save(fig, "01_ecg_snr.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 02 — ECG amplitude · HR · RMSSD · SNR
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(raw, den, t_snr, snr, inverted):
    ds  = 4
    t_e = den["t_ecg"][::ds] / 60
    ecg = den["ecg"][::ds]
    t_s = t_snr / 60
    v_h = ~np.isnan(raw["hr"])
    v_r = ~np.isnan(raw["rmssd"])
    v_s = ~np.isnan(snr)

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1, 1], "hspace": 0.07},
    )
    fig.suptitle("ECG Amplitude · Heart Rate · RMSSD · SNR",
                 fontweight="bold", fontsize=12)

    axes[0].plot(t_e, ecg, lw=0.4, color="#2980b9", label="ECG (denoised)")
    axes[0].set_ylim(-2, 2)
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(raw["t_hr"][v_h] / 60, raw["hr"][v_h],
                 lw=0.9, color="#e74c3c", label="HR")
    axes[1].set_ylabel("HR (bpm)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(raw["t_rmssd"][v_r] / 60, raw["rmssd"][v_r],
                 lw=0.9, color="#27ae60", label="RMSSD")
    axes[2].set_ylabel("RMSSD (ms)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.25)

    # shade inverted regions on all panels
    for ax in axes:
        for t_c, inv in zip(t_s, inverted):
            if inv:
                ax.axvspan(t_c - SNR_STEP_S / 60 / 2,
                           t_c + SNR_STEP_S / 60 / 2,
                           color="#9b59b6", alpha=0.18, lw=0)

    axes[3].plot(t_s[v_s], snr[v_s], lw=1.1, color="#e67e22", label="SNR")
    axes[3].fill_between(t_s[v_s], snr[v_s], snr[v_s].min(),
                         alpha=0.18, color="#e67e22")
    axes[3].set_ylabel("SNR (dB)")
    axes[3].set_xlabel("Time (min)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(alpha=0.25)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    save(fig, "02_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 03 — Accelerometer X / Y / Z / magnitude
# ══════════════════════════════════════════════════════════════════════════════

def plot_acc(raw):
    t = raw["t_acc"] / 60

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.suptitle("Accelerometer — X · Y · Z · Magnitude",
                 fontweight="bold", fontsize=12)

    pairs = [
        (raw["acc_x"], "#e74c3c", "X (LSB)"),
        (raw["acc_y"], "#27ae60", "Y (LSB)"),
        (raw["acc_z"], "#2980b9", "Z (LSB)"),
        (raw["acc_m"], "#8e44ad", "|ACC| (LSB)"),
    ]
    for ax, (data, col, lbl) in zip(axes, pairs):
        ax.plot(t, data, lw=0.55, color=col, label=lbl)
        ax.set_ylabel(lbl, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (min)")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    save(fig, "03_acc.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 04 — ECG raw + denoised + ACC synopsis (first 30 s)
# ══════════════════════════════════════════════════════════════════════════════

def plot_synopsis(raw, den, window_s: float = 30.0):
    def cut(t, v):
        m = t < window_s
        return t[m], v[m]

    fig, axes = plt.subplots(
        5, 1, figsize=(13, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1, 1, 1], "hspace": 0.08},
    )
    fig.suptitle(f"ECG & Accelerometer Synopsis — first {window_s:.0f} s",
                 fontweight="bold", fontsize=12)

    t, v = cut(raw["t_ecg"], raw["ecg"])
    axes[0].plot(t, v, lw=0.6, color="#e74c3c", label="ECG raw")
    axes[0].set_ylabel("µV")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.25)

    t, v = cut(den["t_ecg"], den["ecg"])
    axes[1].plot(t, v, lw=0.6, color="#2980b9", label="ECG denoised")
    axes[1].set_ylabel("µV")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    for ax, (data, col, lbl) in zip(axes[2:], [
        (raw["acc_x"], "#e74c3c", "ACC X"),
        (raw["acc_y"], "#27ae60", "ACC Y"),
        (raw["acc_z"], "#2980b9", "ACC Z"),
    ]):
        t, v = cut(raw["t_acc"], data)
        ax.plot(t, v, lw=0.7, color=col, label=lbl)
        ax.set_ylabel("LSB")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    save(fig, "04_ecg_acc_synopsis.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 05 — 5 raw-ECG windows spanning SNR range (non-inverted only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_snr_examples(raw, t_snr, snr, starts, inverted):
    # work only with valid, non-inverted windows
    valid   = ~np.isnan(snr) & ~inverted
    v_snr   = snr[valid]
    v_start = starts[valid]
    v_t     = t_snr[valid]

    order  = np.argsort(v_snr)
    mid    = len(order) // 2
    # 2 worst, 1 median, 2 best — deduplicate
    indices, seen = [], set()
    for i in [order[0], order[1], order[mid], order[-2], order[-1]]:
        if i not in seen:
            seen.add(i)
            indices.append(i)

    labels = ["Worst SNR", "2nd Worst", "Median SNR", "2nd Best", "Best SNR"]
    colors = ["#c0392b", "#e67e22", "#f39c12", "#2ecc71", "#27ae60"]
    median_snr = np.nanmedian(snr)

    n_inv   = inverted.sum()
    n_total = len(snr)

    fig, axes = plt.subplots(5, 1, figsize=(13, 13))
    fig.suptitle(
        f"Raw ECG — 5 Windows Across SNR Range  ({SNR_WINDOW_S:.0f} s each)  "
        f"[{n_inv}/{n_total} windows excluded as inverted]",
        fontweight="bold", fontsize=12,
    )

    for ax, idx, lbl, col in zip(axes, indices, labels, colors):
        s      = int(v_start[idx])
        seg    = raw["ecg"][s : s + int(SNR_WINDOW_S * ECG_FS)]
        t_win  = np.arange(len(seg)) / ECG_FS
        db     = v_snr[idx]
        offset = v_t[idx] / 60
        qual   = "GOOD" if db >= median_snr else "BAD"

        ax.plot(t_win, seg, lw=0.55, color=col)
        ax.set_ylabel("µV", fontsize=9)
        ax.grid(alpha=0.22)

        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2.0)

        ax.set_title(
            f"{lbl}   SNR = {db:.1f} dB  [{qual}]   "
            f"@ {offset:.1f} min into recording",
            fontsize=9, loc="left", color=col, fontweight="bold", pad=4,
        )

    axes[-1].set_xlabel("Time within window (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "05_snr_examples.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data…")
    raw = load(RAW_PATH)
    den = load(DEN_PATH)

    valid_hr = raw["hr"][~np.isnan(raw["hr"])]
    print(f"  Duration    : {raw['t_ecg'][-1] / 60:.1f} min")
    print(f"  ECG samples : {len(raw['ecg']):,}  ({ECG_FS} Hz)")
    print(f"  ACC samples : {len(raw['acc_x']):,}  ({ACC_FS} Hz)")
    print(f"  HR range    : {int(np.nanmin(valid_hr))}–{int(np.nanmax(valid_hr))} bpm")

    print(f"\nComputing ECG features  "
          f"({SNR_WINDOW_S:.0f} s window, {SNR_STEP_S:.0f} s step)…",
          end=" ", flush=True)
    t_snr, features, starts = compute_windows(raw["ecg"])
    # feature columns: [hr_mean, hr_max, hr_min, hrv, snr, flatline, inverted]
    snr      = features[:, 4]
    inverted = features[:, 6].astype(bool)
    n_inv    = int(np.nansum(inverted))
    n_flat   = int(np.nansum(features[:, 5] == 1.0))
    print(f"done  —  {len(snr)} windows  |  "
          f"{n_inv} inverted  |  {n_flat} flatline")

    print("\nGenerating plots…")
    plot_ecg_snr(den, t_snr, snr, inverted)
    plot_overview(raw, den, t_snr, snr, inverted)
    plot_acc(raw)
    plot_synopsis(raw, den)
    plot_snr_examples(raw, t_snr, snr, starts, inverted)

    print(f"\nDone. All plots saved to: {OUT_DIR}")

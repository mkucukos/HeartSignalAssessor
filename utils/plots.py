from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.vq import kmeans, vq


def save_snr_cluster_plot(
    df: pd.DataFrame,
    output_path: str = "snr_cluster_plot.png",
) -> None:
    """Save a 3-panel scatter plot showing how HR and HRV drift as SNR degrades.

    The SNR quality boundary is determined automatically by K-means (k=2)
    clustering in the HR-HRV feature space. The threshold is the SNR midpoint
    between the two clusters — no hardcoded value.

    Panels
    ------
    1. SNR vs HR mean  — vertical boundary at data-driven threshold
    2. SNR vs HRV      — same boundary
    3. HR vs HRV       — coloured by SNR; cluster annotations show drift
    """
    valid = df[df["features_valid"] & df["hr_mean"].notna() & df["hrv"].notna()].copy()
    if len(valid) == 0:
        print("No valid features to plot.")
        return

    snr = valid["snr"].to_numpy()
    hr  = valid["hr_mean"].to_numpy()
    hrv = valid["hrv"].to_numpy()

    # --- Data-driven SNR threshold via K-means on HR-HRV space ---
    threshold, labels, bad_cluster = _find_snr_threshold(snr, hr, hrv)
    good_cluster = 1 - bad_cluster
    print(f"Data-driven SNR threshold: {threshold:.2f} dB")

    # Baseline: median of the good cluster
    good_mask = labels == good_cluster
    bad_mask  = labels == bad_cluster
    baseline_hr  = np.median(hr[good_mask])
    baseline_hrv = np.median(hrv[good_mask])

    # Colour map: red (low SNR) → green (high SNR)
    snr_min = max(snr.min(), 0)
    snr_max = min(snr.max(), 15)
    norm = mcolors.Normalize(vmin=snr_min, vmax=snr_max)
    cmap = plt.cm.RdYlGn

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        f"SNR vs HR & HRV — Cluster Drift Analysis"
        f"  |  Data-driven threshold: {threshold:.2f} dB",
        fontsize=13, fontweight="bold",
    )

    scatter_kw = dict(c=snr, cmap=cmap, norm=norm, s=40, alpha=0.8, edgecolors="none")

    # --- Panel 1: SNR vs HR ---
    axs[0].scatter(snr, hr, **scatter_kw)
    axs[0].axvline(threshold,   color="black",    linestyle="--", linewidth=1.4,
                   label=f"Threshold = {threshold:.2f} dB")
    axs[0].axhline(baseline_hr, color="steelblue", linestyle="--", linewidth=1.2,
                   label=f"Baseline HR = {baseline_hr:.1f} BPM")
    axs[0].set_xlabel("SNR (dB)")
    axs[0].set_ylabel("HR Mean (BPM)")
    axs[0].set_title("SNR vs Mean Heart Rate")
    axs[0].set_xlim(snr_min - 0.5, snr_max + 0.5)
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)
    _shade_region(axs[0], snr_min - 0.5, threshold)

    # --- Panel 2: SNR vs HRV ---
    axs[1].scatter(snr, hrv, **scatter_kw)
    axs[1].axvline(threshold,    color="black",    linestyle="--", linewidth=1.4,
                   label=f"Threshold = {threshold:.2f} dB")
    axs[1].axhline(baseline_hrv, color="steelblue", linestyle="--", linewidth=1.2,
                   label=f"Baseline HRV = {baseline_hrv:.1f} ms")
    axs[1].set_xlabel("SNR (dB)")
    axs[1].set_ylabel("HRV — RMSSD-like (ms)")
    axs[1].set_title("SNR vs Heart Rate Variability")
    axs[1].set_xlim(snr_min - 0.5, snr_max + 0.5)
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.3)
    _shade_region(axs[1], snr_min - 0.5, threshold)

    # --- Panel 3: HR vs HRV cluster coloured by SNR ---
    sc = axs[2].scatter(hr, hrv, **scatter_kw)
    axs[2].axvline(baseline_hr,  color="steelblue",  linestyle="--", linewidth=1.2,
                   label=f"Baseline HR = {baseline_hr:.1f} BPM")
    axs[2].axhline(baseline_hrv, color="darkorange", linestyle="--", linewidth=1.2,
                   label=f"Baseline HRV = {baseline_hrv:.1f} ms")

    # Annotate cluster centres
    if bad_mask.any():
        cx, cy = np.median(hr[bad_mask]), np.median(hrv[bad_mask])
        axs[2].annotate(
            f"Poor signal\n(SNR < {threshold:.1f} dB)", xy=(cx, cy),
            fontsize=8, color="darkred", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8),
        )
    if good_mask.any():
        cx, cy = np.median(hr[good_mask]), np.median(hrv[good_mask])
        axs[2].annotate(
            f"Good signal\n(SNR ≥ {threshold:.1f} dB)", xy=(cx, cy),
            fontsize=8, color="darkgreen", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", alpha=0.8),
        )

    axs[2].set_xlabel("HR Mean (BPM)")
    axs[2].set_ylabel("HRV — RMSSD-like (ms)")
    axs[2].set_title("HR vs HRV Cluster  (colour = SNR)")
    axs[2].legend(fontsize=8)
    axs[2].grid(True, alpha=0.3)

    # Shared colourbar
    cbar = fig.colorbar(sc, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("SNR (dB)", fontsize=10)
    cbar.ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cluster plot saved as '{output_path}'")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_snr_threshold(
    snr: np.ndarray,
    hr: np.ndarray,
    hrv: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """K-means (k=2) on normalised HR-HRV space → SNR boundary.

    Returns
    -------
    threshold : float
        Midpoint SNR between the highest-SNR bad-cluster point and
        the lowest-SNR good-cluster point.
    labels : np.ndarray
        Cluster label (0 or 1) per frame.
    bad_cluster : int
        Which label index corresponds to the low-SNR cluster.
    """
    hr_norm  = (hr  - hr.mean())  / (hr.std()  + 1e-9)
    hrv_norm = (hrv - hrv.mean()) / (hrv.std() + 1e-9)
    features = np.column_stack([hr_norm, hrv_norm]).astype(np.float64)

    centroids, _ = kmeans(features, 2, seed=42)
    labels, _    = vq(features, centroids)

    snr_means  = [snr[labels == i].mean() for i in range(2)]
    bad_cluster = int(np.argmin(snr_means))
    good_cluster = 1 - bad_cluster

    bad_snr_max  = snr[labels == bad_cluster].max()
    good_snr_min = snr[labels == good_cluster].min()
    threshold = (bad_snr_max + good_snr_min) / 2.0

    return threshold, labels, bad_cluster


def _shade_region(ax, x_start: float, x_end: float) -> None:
    ax.axvspan(x_start, x_end, color="red", alpha=0.07)

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes


_FS = 250  # samples per second — mirrors the pipeline default


def create_animation(
    df: pd.DataFrame,
    valid_features: pd.DataFrame,
    output_stem: str = "ecg_analysis_animation",
) -> None:
    """Render and save the 8-panel real-time ECG animation.

    Parameters
    ----------
    df : pd.DataFrame
        Full frame data returned by generate_ecg_data().
    valid_features : pd.DataFrame
        Valid-feature subset with a 'prediction' column, returned by generate_ecg_data().
    output_stem : str
        Base filename for the output (without extension).
        MP4 is attempted first; GIF is the fallback.
    """
    print("Creating animation...")

    # Mutable accumulator — updated each frame inside the closure
    state: dict[str, list] = {
        "frames": [], "times": [],
        "hr_mean": [], "hr_max": [], "hr_min": [],
        "hrv": [], "snr": [],
        "ecg_data": [], "ecg_times": [],
        "predictions": [], "pred_times": [],
    }

    fig, axs = plt.subplots(8, 1, figsize=(12, 12))

    def _draw_frame(frame_idx: int) -> None:
        if frame_idx >= len(df):
            return

        row = df.iloc[frame_idx]
        for ax in axs:
            ax.clear()

        ecg_plot = row["ecg_plot"]

        # Panel 0 — current ECG window (10-second tail)
        if ecg_plot is not None and len(ecg_plot) > 0:
            axs[0].plot(row["time_plot"], ecg_plot, "k-", linewidth=0.8)
            axs[0].set_title(
                f"Current ECG Window — Frame {row['frame']}  |  Noise STD: {row['noise_std']:.3f}"
            )
        else:
            axs[0].set_title(f"ECG Signal — Frame {row['frame']}  |  Buffer filling...")
        _style(axs[0], ylabel="Amplitude (mV)")

        # Accumulate state for valid frames
        if row["features_valid"] and not pd.isna(row["hr_mean"]):
            for key in ("hr_mean", "hr_max", "hr_min", "hrv", "snr"):
                state[key].append(row[key])
            state["frames"].append(row["frame"])
            state["times"].append(row["time_global"])

            if ecg_plot is not None and len(ecg_plot) > 0:
                dt = 1 / _FS
                if not state["ecg_times"]:
                    state["ecg_times"] = [i * dt for i in range(len(ecg_plot))]
                else:
                    last = state["ecg_times"][-1]
                    state["ecg_times"].extend(
                        [last + (i + 1) * dt for i in range(len(ecg_plot))]
                    )
                state["ecg_data"].extend(ecg_plot)

            # Attach prediction for this frame if available
            match = valid_features[valid_features["frame"] == row["frame"]]
            if len(match) > 0 and "prediction" in match.columns:
                pred_val = match.iloc[0]["prediction"]
                if not pd.isna(pred_val):
                    state["predictions"].append(pred_val)
                    state["pred_times"].append(row["time_global"])

        # Panel 1 — cumulative ECG timeline with highlighted current segment
        if state["ecg_data"]:
            axs[1].plot(state["ecg_times"], state["ecg_data"], "k-", linewidth=0.8)
            if (
                row["features_valid"]
                and not pd.isna(row["hr_mean"])
                and ecg_plot is not None
                and len(ecg_plot) > 0
            ):
                seg_len = len(ecg_plot)
                if len(state["ecg_times"]) >= seg_len:
                    axs[1].plot(
                        state["ecg_times"][-seg_len:],
                        state["ecg_data"][-seg_len:],
                        "red", linewidth=1.2, alpha=0.8,
                    )
            total_dur = len(state["ecg_data"]) / _FS
            axs[1].set_title(f"Cumulative ECG — Total: {total_dur:.1f}s")
            axs[1].set_xlim(state["ecg_times"][0], state["ecg_times"][-1])
        else:
            n_valid = len(state["times"])
            axs[1].text(
                0.5, 0.5,
                f"Waiting for valid features...\n({n_valid} valid frames so far)",
                ha="center", va="center", transform=axs[1].transAxes, fontsize=10,
            )
            axs[1].set_title("Cumulative ECG Signal")
        _style(axs[1], ylabel="Amplitude (mV)")

        # Panels 2-6 — feature time series
        if state["times"]:
            t = state["times"]
            _plot_series(axs[2], t, state["snr"],     "purple", "SNR (dB)",  "Signal-to-Noise Ratio (SNR)", ylim=(0, 30))
            _plot_series(axs[3], t, state["hr_mean"], "blue",   "HR (BPM)", "Mean Heart Rate",              hline=70)
            _plot_series(axs[4], t, state["hr_max"],  "red",    "HR (BPM)", "Maximum Heart Rate")
            _plot_series(axs[5], t, state["hr_min"],  "green",  "HR (BPM)", "Minimum Heart Rate")
            _plot_series(axs[6], t, state["hrv"],     "black",  "HRV (ms)", "Heart Rate Variability (RMSSD)")

        # Panel 7 — ML prediction probability
        if state["predictions"]:
            preds = state["predictions"]
            pred_t = state["pred_times"]
            axs[7].plot(pred_t, preds, "black", linewidth=2)
            axs[7].axhline(y=0.5, linestyle="--", color="red", alpha=0.7)
            axs[7].fill_between(
                pred_t, preds,
                where=(np.array(preds) >= 0.5),
                color="red", alpha=0.3,
            )
            axs[7].set_title(f"Model Prediction Probability  ({len(preds)} predictions)")
        else:
            n_feat = len(state["times"])
            msg = (
                f"Waiting for predictions...\n({n_feat} features so far)"
                if n_feat > 0
                else "Waiting for valid features..."
            )
            axs[7].text(
                0.5, 0.5, msg,
                ha="center", va="center", transform=axs[7].transAxes, fontsize=10,
            )
            axs[7].set_title("Model Prediction Probability")
        axs[7].set_ylabel("Probability")
        axs[7].set_ylim(0, 1)
        axs[7].grid(True, alpha=0.3)

        axs[-1].set_xlabel("Time (s)")

        ecg_dur = len(state["ecg_data"]) / _FS if state["ecg_data"] else 0.0
        fig.suptitle(
            f"Real-time ECG Analysis — Frame {frame_idx + 1}/{len(df)}"
            f"  |  Cumulative ECG: {ecg_dur:.1f}s"
            f"  |  Features: {len(state['times'])}"
            f"  |  Predictions: {len(state['predictions'])}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

    anim = animation.FuncAnimation(
        fig, _draw_frame,
        frames=len(df),
        interval=50,
        blit=False,
        repeat=False,
    )

    _save(anim, output_stem)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _style(ax: Axes, ylabel: str = "") -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _plot_series(
    ax: Axes,
    x: list,
    y: list,
    color: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    hline: float | None = None,
) -> None:
    ax.plot(x, y, color, linewidth=2)
    ax.set_title(title)
    _style(ax, ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if hline is not None:
        ax.axhline(y=hline, color="red", linestyle="--", alpha=0.5)


def _save(anim: animation.FuncAnimation, stem: str) -> None:
    try:
        writer = animation.writers["ffmpeg"](
            fps=25, metadata={"artist": "ECG Analyzer"}, bitrate=1800
        )
        path = f"{stem}.mp4"
        anim.save(path, writer=writer)
        print(f"Animation saved as '{path}'")
        return
    except Exception:
        pass

    try:
        path = f"{stem}.gif"
        anim.save(path, writer="pillow", fps=5)
        print(f"Animation saved as '{path}'")
    except Exception as exc:
        print(f"Could not save animation: {exc}")
        plt.show()

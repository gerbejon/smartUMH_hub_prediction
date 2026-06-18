"""
Shared plotting utilities for the forecasting pipelines
========================================================
Centralises the train / test / forecast plot so each model pipeline can import
it instead of redefining `plot_results` locally.

Usage
-----
    from plot_utils import plot_results

    plot_results(series_train, series_test, forecast_df,
                 model_name="naive", postfix="hub1")

The forecast frame is expected to follow the shared pipeline format:
    columns = ["forecast", "lower_ci", "upper_ci"]  (CI columns optional)
    index   = series_test.index
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default output folder for plots: <project>/zur_sim/plots (cwd-independent).
# This file lives at <project>/zur_sim/models/plot_utils.py
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def plot_results(
    series_train,
    series_test,
    forecast_df,
    model_name: str = "model",
    postfix: str = "",
    title: str | None = None,
    plot_short: bool = False,
    save: bool = True,
    show: bool = False,
    out_dir=None,
):
    """
    Plot training history, true test values and the forecast (+ 95% CI band
    when the frame carries lower_ci / upper_ci columns).

    Parameters
    ----------
    series_train, series_test : pd.Series
        Observed train / test portions of the series.
    forecast_df : pd.DataFrame
        Forecast frame with a "forecast" column (and optional CI columns),
        indexed like series_test.
    model_name : str
        Used for the default title and the output file name.
    postfix : str
        Suffix for the output file name (e.g. "hub1").
    title : str, optional
        Overrides the default "<model_name> — Train / Test Forecast" title.
    plot_short : bool
        If True, only show the last 20% of the training history.
    save : bool
        If True (default), write "<model_name>_forecast_<postfix>.png" into
        `out_dir` (defaults to PLOTS_DIR = <project>/zur_sim/plots).
    show : bool
        If True, call plt.show().
    out_dir : str | Path, optional
        Folder to save into; defaults to PLOTS_DIR. Created if missing.
    """
    if title is None:
        title = f"{model_name} — Train / Test Forecast"

    if plot_short:
        series_train = series_train.iloc[round(len(series_train) * 0.8):]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(series_train.index, series_train.values,
            label="Train (observed)", color="#2563eb", linewidth=1.5)
    ax.plot(series_test.index, series_test.values,
            label="Test (true)", color="#16a34a", linewidth=1.5)
    ax.plot(forecast_df.index, forecast_df["forecast"],
            label="Forecast", color="#dc2626", linewidth=2, linestyle="--")

    if "lower_ci" in forecast_df and "upper_ci" in forecast_df:
        ax.fill_between(
            forecast_df.index,
            forecast_df["lower_ci"],
            forecast_df["upper_ci"],
            color="#dc2626", alpha=0.15, label="95% CI",
        )

    ax.axvline(series_test.index[0], color="gray", linestyle=":",
               linewidth=1.5, label="Train/test split")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"{model_name}_forecast_{postfix}.png" if postfix else f"{model_name}_forecast.png"
    out_file = None
    if save:
        out_dir = Path(out_dir) if out_dir is not None else PLOTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / fname
        plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    print(f"[Plot] {'saved ' + str(out_file) if out_file else fname + ' (not saved)'}")
    return fig, ax


# ─────────────────────────────────────────────
# MULTI-MODEL COMPARISON PLOTS
# ─────────────────────────────────────────────
# These compare several models on a single held-out test window. They expect:
#   predictions : dict {model_name: np.ndarray of len(test)}
#   metrics_df  : pd.DataFrame with columns ["model", "MAE", "RMSE", "sMAPE_%"]
# and save the figure to `out_path` (no plt.show, so they work in batch runs).


def _model_colors(names):
    """Stable color per model so legends/charts agree across figures."""
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names)}


def plot_overview(df, train, test, out_path, target: str = ""):
    """Full series with the 80/20 train/test split highlighted."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(train.index, train["y"], color="#1f77b4", lw=0.8, label=f"train ({len(train)}h)")
    ax.plot(test.index, test["y"], color="#d62728", lw=0.8, label=f"test ({len(test)}h)")
    ax.axvspan(test.index[0], test.index[-1], color="#d62728", alpha=0.07)
    ax.set_title(f"Hub {target} share — full series with 80/20 split")
    ax.set_ylabel("share (pp)")
    ax.set_xlabel("time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_forecasts_combined(train, test, predictions, out_path, target: str = ""):
    """All model forecasts overlaid on the test window (+ last 168h of train)."""
    colors = _model_colors(predictions.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    tail = train.iloc[-168:]
    ax.plot(tail.index, tail["y"], color="black", lw=1, label="train (last 168h)")
    ax.plot(test.index, test["y"], color="black", lw=1.8, label="test (actual)")
    for name, yhat in predictions.items():
        ax.plot(test.index, yhat, lw=1.0, alpha=0.85, label=name, color=colors[name])
    ax.set_title(f"Hub {target} share — multi-step forecast on held-out test window")
    ax.set_ylabel("share (pp)")
    ax.set_xlabel("time")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_forecasts_grid(test, predictions, metrics_df, out_path, target: str = ""):
    """One subplot per model: forecast vs actual, annotated with its metrics."""
    colors = _model_colors(predictions.keys())
    n = len(predictions)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    metric_lookup = metrics_df.set_index("model")
    for ax, (name, yhat) in zip(axes, predictions.items()):
        ax.plot(test.index, test["y"], color="black", lw=1.4, label="actual")
        ax.plot(test.index, yhat, color=colors[name], lw=1.2, label="forecast")
        m = metric_lookup.loc[name]
        ax.set_title(f"{name}   MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  sMAPE={m['sMAPE_%']:.1f}%",
                     fontsize=10)
        ax.set_ylabel("share (pp)")
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Hub {target} share — per-model forecast vs actual", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_metrics_bar(metrics_df, out_path, target: str = ""):
    """Horizontal bar chart comparing MAE / RMSE / sMAPE across models."""
    metric_cols = ["MAE", "RMSE", "sMAPE_%"]
    df = metrics_df.set_index("model")[metric_cols].sort_values("RMSE")
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(14, 4))
    for ax, col in zip(axes, metric_cols):
        bars = ax.barh(df.index[::-1], df[col][::-1], color="#4c78a8")
        ax.set_title(col)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(right=df[col].max() * 1.18)
    fig.suptitle(f"Hub {target} share — model comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_residuals(test, predictions, out_path):
    """Per-model residual time series + residual distribution."""
    colors = _model_colors(predictions.keys())
    n = len(predictions)
    fig, axes = plt.subplots(n, 2, figsize=(14, 1.8 * n), sharex="col")
    axes = np.atleast_2d(axes)
    y_true = test["y"].to_numpy()
    for i, (name, yhat) in enumerate(predictions.items()):
        resid = yhat - y_true
        ax_ts, ax_hist = axes[i, 0], axes[i, 1]
        ax_ts.axhline(0, color="black", lw=0.5)
        ax_ts.plot(test.index, resid, color=colors[name], lw=0.9)
        ax_ts.set_ylabel(name, fontsize=9)
        ax_ts.grid(alpha=0.25)
        ax_hist.hist(resid, bins=25, color=colors[name], alpha=0.85)
        ax_hist.axvline(0, color="black", lw=0.5)
        ax_hist.grid(alpha=0.25)
    axes[0, 0].set_title("residual over time (forecast − actual, pp)")
    axes[0, 1].set_title("residual distribution")
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("residual (pp)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_error_by_hour(test, predictions, out_path, target: str = ""):
    """Mean absolute error by hour-of-day for each model."""
    colors = _model_colors(predictions.keys())
    hours = test.index.hour
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, yhat in predictions.items():
        abs_err = np.abs(yhat - test["y"].to_numpy())
        by_hour = pd.Series(abs_err, index=hours).groupby(level=0).mean()
        ax.plot(by_hour.index, by_hour.values, marker="o", lw=1.2,
                label=name, color=colors[name])
    ax.set_title(f"Hub {target} share — mean |error| by hour of day (test window)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("MAE (pp)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


# ─────────────────────────────────────────────
# WALK-FORWARD (ROLLING-ORIGIN) PLOTS
# ─────────────────────────────────────────────
# These summarise an expanding-window walk-forward evaluation. They expect:
#   per_origin : pd.DataFrame with columns ["model", "cutoff", "MAE", "RMSE", "sMAPE_%", ...]
#   summary    : pd.DataFrame with columns ["model", "<METRIC>_mean", "<METRIC>_std", ...]


def plot_wf_per_origin(per_origin, metric, out_path):
    """Per-origin metric trace, one line per model."""
    colors = _model_colors(sorted(per_origin["model"].unique()))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for name, sub in per_origin.groupby("model"):
        sub = sub.sort_values("cutoff")
        ax.plot(sub["cutoff"], sub[metric], marker="o", lw=1.1, ms=4,
                label=name, color=colors[name])
    ax.set_title(f"Walk-forward {metric} per origin")
    ax.set_xlabel("origin cutoff (last train timestamp)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_wf_box(per_origin, metric, out_path):
    """Distribution of a metric across origins, one box per model."""
    order = (per_origin.groupby("model")[metric].mean().sort_values().index.tolist())
    data = [per_origin.loc[per_origin["model"] == m, metric].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 5})
    colors = _model_colors(order)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax.set_title(f"Walk-forward {metric} distribution across origins")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_wf_summary_bars(summary, out_path, target: str = ""):
    """Mean ± std bars per model for MAE / RMSE / sMAPE across origins."""
    pairs = [("MAE_mean", "MAE_std"), ("RMSE_mean", "RMSE_std"), ("sMAPE_mean", "sMAPE_std")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (mean_col, std_col) in zip(axes, pairs):
        df = summary.sort_values(mean_col)
        ax.barh(df["model"], df[mean_col], xerr=df[std_col], color="#4c78a8",
                error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"})
        for i, (m, s) in enumerate(zip(df[mean_col], df[std_col])):
            ax.text(m + s + df[mean_col].max() * 0.01, i, f"{m:.2f}±{s:.2f}",
                    va="center", fontsize=8)
        ax.set_title(mean_col.replace("_mean", ""))
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(right=(df[mean_col] + df[std_col]).max() * 1.25)
    fig.suptitle(f"Hub {target} share — walk-forward mean ± std across origins")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")

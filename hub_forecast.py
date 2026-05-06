"""Forecast hub Z047 share for the Feb 2026 hub_distribution series.

Pipeline:
    load -> chronological 80/20 split -> baselines + SARIMA + XGBoost
        -> metrics (MAE / RMSE / sMAPE / MAE-pp) -> plot.

All models produce a single multi-step forecast over the test horizon using
only training data — no test leakage. XGBoost forecasts recursively, feeding
its own predictions into the lag features at each step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "zur_sim" / "data" / "hub_distribution.csv"
PLOT_DIR = ROOT / "plots"

TARGET = "Z047"
TRAIN_FRAC = 0.8


# ----------------------------- data ---------------------------------------- #

def load_hub_series(path: Path = DATA_PATH, target: str = TARGET) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["t"]).set_index("t").asfreq("h")
    # Z047 + Z059 sum to 100 — fill missing Z059 values from Z047.
    df["Z059"] = df["Z059"].fillna(100.0 - df["Z047"])
    df["Z047"] = df["Z047"].astype(float)
    return df[[target]].rename(columns={target: "y"})


def chrono_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ----------------------------- metrics ------------------------------------- #

def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(100.0 * np.mean(np.where(denom == 0, 0.0, 2.0 * np.abs(err) / denom)))
    # y is already expressed in percentage points, so MAE itself is in pp.
    return {"MAE": mae, "RMSE": rmse, "sMAPE_%": smape, "MAE_pp": mae}


# ----------------------------- baselines ----------------------------------- #

def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train["y"].iloc[-1], dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame, season: int) -> np.ndarray:
    last_season = train["y"].iloc[-season:].to_numpy()
    reps = int(np.ceil(len(test) / season))
    return np.tile(last_season, reps)[: len(test)]


def moving_average_forecast(train: pd.DataFrame, test: pd.DataFrame, window: int = 24) -> np.ndarray:
    return np.full(len(test), train["y"].iloc[-window:].mean(), dtype=float)


# ----------------------------- SARIMA -------------------------------------- #

def sarima_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
) -> np.ndarray:
    model = sm.tsa.statespace.SARIMAX(
        train["y"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    return res.forecast(steps=len(test)).to_numpy()


# ----------------------------- XGBoost ------------------------------------- #

LAGS = [1, 2, 3, 24, 25, 48, 168, 169]
ROLL_WINDOWS = [24, 168]
FEATURE_COLS = (
    [f"lag{l}" for l in LAGS]
    + [f"roll{w}_mean" for w in ROLL_WINDOWS]
    + ["hour", "dow", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
)


def calendar_features(ts: pd.Timestamp) -> dict:
    return {
        "hour": ts.hour,
        "dow": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
    }


def build_training_frame(train: pd.DataFrame) -> pd.DataFrame:
    s = train["y"]
    feats = {f"lag{l}": s.shift(l) for l in LAGS}
    for w in ROLL_WINDOWS:
        feats[f"roll{w}_mean"] = s.shift(1).rolling(w).mean()
    df = pd.DataFrame(feats, index=train.index)
    df["hour"] = train.index.hour
    df["dow"] = train.index.dayofweek
    df["is_weekend"] = (train.index.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * train.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * train.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * train.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * train.index.dayofweek / 7)
    df["y"] = s
    return df.dropna()


def feature_row(history: pd.Series, ts: pd.Timestamp) -> np.ndarray:
    # history is indexed by hour and contains all values strictly before `ts`.
    vals = {f"lag{l}": history.iloc[-l] for l in LAGS}
    for w in ROLL_WINDOWS:
        vals[f"roll{w}_mean"] = history.iloc[-w:].mean()
    vals.update(calendar_features(ts))
    return np.array([vals[c] for c in FEATURE_COLS], dtype=float)


def xgboost_recursive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train_df = build_training_frame(train)
    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["y"].to_numpy()

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    history = train["y"].copy()
    preds = np.empty(len(test), dtype=float)
    for i, ts in enumerate(test.index):
        x = feature_row(history, ts).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        # clip to physical range (share is bounded to [0, 100]).
        yhat = float(np.clip(yhat, 0.0, 100.0))
        preds[i] = yhat
        history = pd.concat([history, pd.Series([yhat], index=[ts])])
    return preds


# ----------------------------- plot ---------------------------------------- #

# Stable color per model so legends/charts agree across figures.
def _model_colors(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names)}


def plot_overview(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(train.index, train["y"], color="#1f77b4", lw=0.8, label=f"train ({len(train)}h)")
    ax.plot(test.index, test["y"], color="#d62728", lw=0.8, label=f"test ({len(test)}h)")
    ax.axvspan(test.index[0], test.index[-1], color="#d62728", alpha=0.07)
    ax.set_title(f"Hub {TARGET} share — full Feb 2026 series with 80/20 split")
    ax.set_ylabel("share (pp)")
    ax.set_xlabel("time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_forecasts_combined(train: pd.DataFrame, test: pd.DataFrame, predictions: dict, out_path: Path) -> None:
    colors = _model_colors(predictions.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    tail = train.iloc[-168:]
    ax.plot(tail.index, tail["y"], color="black", lw=1, label="train (last 168h)")
    ax.plot(test.index, test["y"], color="black", lw=1.8, label="test (actual)")
    for name, yhat in predictions.items():
        ax.plot(test.index, yhat, lw=1.0, alpha=0.85, label=name, color=colors[name])
    ax.set_title(f"Hub {TARGET} share — multi-step forecast on held-out test window")
    ax.set_ylabel("share (pp)")
    ax.set_xlabel("time")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_forecasts_grid(test: pd.DataFrame, predictions: dict, metrics_df: pd.DataFrame, out_path: Path) -> None:
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
    fig.suptitle(f"Hub {TARGET} share — per-model forecast vs actual", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    metric_cols = ["MAE", "RMSE", "sMAPE_%"]
    df = metrics_df.set_index("model")[metric_cols].sort_values("RMSE")
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(14, 4))
    for ax, col in zip(axes, metric_cols):
        bars = ax.barh(df.index[::-1], df[col][::-1], color="#4c78a8")
        ax.set_title(col)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(right=df[col].max() * 1.18)
    fig.suptitle(f"Hub {TARGET} share — model comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residuals(test: pd.DataFrame, predictions: dict, out_path: Path) -> None:
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


def plot_error_by_hour(test: pd.DataFrame, predictions: dict, out_path: Path) -> None:
    colors = _model_colors(predictions.keys())
    hours = test.index.hour
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, yhat in predictions.items():
        abs_err = np.abs(yhat - test["y"].to_numpy())
        by_hour = pd.Series(abs_err, index=hours).groupby(level=0).mean()
        ax.plot(by_hour.index, by_hour.values, marker="o", lw=1.2,
                label=name, color=colors[name])
    ax.set_title(f"Hub {TARGET} share — mean |error| by hour of day (test window)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("MAE (pp)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------- walk-forward -------------------------------- #

# Uniform model registry — each callable: (train_df, test_df) -> np.ndarray of len(test)
def model_registry() -> dict:
    return {
        "naive": naive_forecast,
        "seasonal_naive_24h": lambda tr, te: seasonal_naive_forecast(tr, te, 24),
        "seasonal_naive_168h": lambda tr, te: seasonal_naive_forecast(tr, te, 168),
        "moving_avg_24h": lambda tr, te: moving_average_forecast(tr, te, 24),
        "sarima": sarima_forecast,
        "xgboost": xgboost_recursive_forecast,
    }


def walk_forward(
    df: pd.DataFrame,
    models: dict,
    min_train: int = 336,
    horizon: int = 24,
    step: int = 24,
):
    """Expanding-window walk-forward.

    For each origin starting at hour `min_train`, fit on df[:origin] and forecast
    df[origin : origin + horizon]. Returns (per_origin_df, summary_df).
    """
    n = len(df)
    origins = list(range(min_train, n - horizon + 1, step))
    rows = []
    for k, origin in enumerate(origins):
        train = df.iloc[:origin]
        test = df.iloc[origin : origin + horizon]
        cutoff = train.index[-1]
        for name, fn in models.items():
            yhat = np.asarray(fn(train, test), dtype=float)
            m = metrics(test["y"].to_numpy(), yhat)
            rows.append({"origin_idx": k, "cutoff": cutoff, "model": name, **m})
        print(f"  origin {k+1}/{len(origins)}  cutoff={cutoff}")

    per_origin = pd.DataFrame(rows)
    summary = (
        per_origin.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            sMAPE_mean=("sMAPE_%", "mean"), sMAPE_std=("sMAPE_%", "std"),
            n_origins=("MAE", "count"),
        )
        .sort_values("RMSE_mean")
        .reset_index()
    )
    return per_origin, summary


def plot_wf_per_origin(per_origin: pd.DataFrame, metric: str, out_path: Path) -> None:
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


def plot_wf_box(per_origin: pd.DataFrame, metric: str, out_path: Path) -> None:
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


def plot_wf_summary_bars(summary: pd.DataFrame, out_path: Path) -> None:
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
    fig.suptitle(f"Hub {TARGET} share — walk-forward mean ± std across origins")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------- main ---------------------------------------- #

def main() -> None:
    df = load_hub_series()
    train, test = chrono_split(df)
    print(f"target={TARGET}  train={len(train)}h  test={len(test)}h  "
          f"({train.index.min()} -> {train.index.max()} | {test.index.min()} -> {test.index.max()})")

    preds: dict[str, np.ndarray] = {}
    preds["naive"] = naive_forecast(train, test)
    preds["seasonal_naive_24h"] = seasonal_naive_forecast(train, test, 24)
    preds["seasonal_naive_168h"] = seasonal_naive_forecast(train, test, 168)
    preds["moving_avg_24h"] = moving_average_forecast(train, test, 24)
    print("fitting SARIMA(1,0,1)x(1,0,1,24)...")
    preds["sarima"] = sarima_forecast(train, test)
    print("fitting XGBoost (recursive)...")
    preds["xgboost"] = xgboost_recursive_forecast(train, test)

    rows = []
    for name, yhat in preds.items():
        rows.append({"model": name, **metrics(test["y"].to_numpy(), yhat)})
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print()
    print(res.to_string(index=False, float_format="{:.3f}".format))

    PLOT_DIR.mkdir(exist_ok=True)
    metrics_path = PLOT_DIR / f"hub_{TARGET}_metrics.csv"
    res.to_csv(metrics_path, index=False)

    outputs = {
        "overview": PLOT_DIR / f"hub_{TARGET}_overview.png",
        "forecasts": PLOT_DIR / f"hub_{TARGET}_forecasts.png",
        "forecasts_grid": PLOT_DIR / f"hub_{TARGET}_forecasts_grid.png",
        "metrics_bar": PLOT_DIR / f"hub_{TARGET}_metrics_bar.png",
        "residuals": PLOT_DIR / f"hub_{TARGET}_residuals.png",
        "error_by_hour": PLOT_DIR / f"hub_{TARGET}_error_by_hour.png",
    }
    plot_overview(df, train, test, outputs["overview"])
    plot_forecasts_combined(train, test, preds, outputs["forecasts"])
    plot_forecasts_grid(test, preds, res, outputs["forecasts_grid"])
    plot_metrics_bar(res, outputs["metrics_bar"])
    plot_residuals(test, preds, outputs["residuals"])
    plot_error_by_hour(test, preds, outputs["error_by_hour"])

    print(f"\nmetrics -> {metrics_path}")
    for name, p in outputs.items():
        print(f"plot[{name}] -> {p}")


def main_walk_forward(min_train: int = 336, horizon: int = 24, step: int = 24) -> None:
    df = load_hub_series()
    models = model_registry()
    n_origins = len(range(min_train, len(df) - horizon + 1, step))
    print(f"target={TARGET}  total_hours={len(df)}  min_train={min_train}h  "
          f"horizon={horizon}h  step={step}h  origins={n_origins}")

    per_origin, summary = walk_forward(df, models, min_train=min_train,
                                       horizon=horizon, step=step)

    print()
    print("Walk-forward summary (mean ± std across origins, sorted by RMSE_mean):")
    print(summary.to_string(index=False, float_format="{:.3f}".format))

    PLOT_DIR.mkdir(exist_ok=True)
    suffix = f"wf_h{horizon}_s{step}"
    paths = {
        "per_origin_csv":   PLOT_DIR / f"hub_{TARGET}_{suffix}_per_origin.csv",
        "summary_csv":      PLOT_DIR / f"hub_{TARGET}_{suffix}_summary.csv",
        "rmse_per_origin":  PLOT_DIR / f"hub_{TARGET}_{suffix}_rmse_per_origin.png",
        "mae_per_origin":   PLOT_DIR / f"hub_{TARGET}_{suffix}_mae_per_origin.png",
        "rmse_box":         PLOT_DIR / f"hub_{TARGET}_{suffix}_rmse_box.png",
        "summary_bars":     PLOT_DIR / f"hub_{TARGET}_{suffix}_summary_bars.png",
    }
    per_origin.to_csv(paths["per_origin_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    plot_wf_per_origin(per_origin, "RMSE", paths["rmse_per_origin"])
    plot_wf_per_origin(per_origin, "MAE", paths["mae_per_origin"])
    plot_wf_box(per_origin, "RMSE", paths["rmse_box"])
    plot_wf_summary_bars(summary, paths["summary_bars"])

    print()
    for name, p in paths.items():
        print(f"{name} -> {p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hub Z047 share forecasting pipeline")
    parser.add_argument("mode", nargs="?", choices=["split", "walk-forward"],
                        default="split",
                        help="'split' = single 80/20 holdout; 'walk-forward' = rolling-origin eval")
    parser.add_argument("--horizon", type=int, default=24,
                        help="forecast horizon in hours (walk-forward only)")
    parser.add_argument("--step", type=int, default=24,
                        help="hours between successive origins (walk-forward only)")
    parser.add_argument("--min-train", type=int, default=336,
                        help="minimum training size in hours before first origin")
    args = parser.parse_args()

    if args.mode == "split":
        main()
    else:
        main_walk_forward(min_train=args.min_train, horizon=args.horizon, step=args.step)

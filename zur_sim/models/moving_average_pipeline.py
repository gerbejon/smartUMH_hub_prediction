"""
Moving-Average Forecasting Pipeline
====================================
Baseline pipeline that mirrors the structure of sarima_pipeline.py /
ets_pipeline.py so it plugs into calculate_all.py unchanged.

Moving-average forecast:
  - the whole test horizon is predicted as the mean of the last `window`
    training observations (a flat forecast).
  - the window length is chosen automatically: the candidate window with the
    lowest one-step in-sample backtest MAE is selected.
  - confidence band is derived from the std of in-sample one-step residuals.

Reuses the shared layout from the other pipelines:
  - plot_results()  (train | true test | forecast + CI)
  - evaluate()      (MAE, RMSE, MAPE)
  - get_data()      (../data/hub_distribution.csv, two hub columns)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from plot_utils import plot_results

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical layout to sarima_pipeline.py)
# ─────────────────────────────────────────────

def evaluate(series_test: pd.Series, forecast_df: pd.DataFrame) -> dict:
    true = series_test.values
    pred = forecast_df["forecast"].values

    mae  = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mape = np.mean(np.abs((true - pred) / np.where(true == 0, np.nan, true))) * 100

    print("\n[Evaluation on test set]")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def get_data():
    # Resolve ../data relative to this file so it works from any working dir.
    data_path = Path(__file__).resolve().parent.parent / "data" / "hub_distribution.csv"
    df = pd.read_csv(data_path, index_col=0)
    df_hub1 = df.iloc[:, 0].fillna(0)
    df_hub2 = df.iloc[:, 1].fillna(0)
    return df_hub1, df_hub2


# ─────────────────────────────────────────────
# WINDOW SELECTION  (Moving-Average-specific)
# ─────────────────────────────────────────────

def select_window(series_train: pd.Series, candidate_windows: list[int], verbose: bool = True) -> dict:
    """
    Pick the window whose one-step rolling-mean prediction
    (y_hat_t = mean(y_{t-window .. t-1})) has the lowest in-sample MAE.
    """
    s = series_train.astype(float)
    n = len(s)

    best_mae = np.inf
    best_w   = None
    results  = []
    for w in sorted(set(int(c) for c in candidate_windows if 1 <= int(c) < n)):
        pred = s.shift(1).rolling(w).mean()
        err  = (s - pred).dropna()
        mae  = float(np.mean(np.abs(err))) if len(err) else np.inf
        results.append({"window": w, "in_sample_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_w   = w
        if verbose:
            print(f"  window={w:4d} -> in-sample MAE={mae:.4f}")

    if best_w is None:  # degenerate fallback
        best_w, best_mae = max(1, n // 2), float("nan")

    print(f"\n[Best Window] window={best_w}  in-sample MAE={best_mae:.4f}")
    return {"window": best_w, "in_sample_mae": best_mae}


# ─────────────────────────────────────────────
# FORECAST  (Moving-Average-specific)
# ─────────────────────────────────────────────

def forecast_moving_average(
    series_train: pd.Series,
    series_test: pd.Series,
    window: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Predict every test step as the mean of the last `window` training values.
    The 95% band uses the std of in-sample one-step rolling-mean residuals.
    """
    mean_val = float(series_train.iloc[-window:].mean())
    forecast = np.full(len(series_test), mean_val, dtype=float)

    s         = series_train.astype(float)
    resid     = (s - s.shift(1).rolling(window).mean()).dropna()
    resid_std = float(resid.std()) if len(resid) else 0.0
    z = 1.96  # ~95% normal band (alpha=0.05)
    band = z * resid_std

    return pd.DataFrame({
        "forecast": forecast,
        "lower_ci": forecast - band,
        "upper_ci": forecast + band,
    }, index=series_test.index)


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def moving_average_pipeline(
    series: pd.Series,
    max_period: int = 365,
    candidate_windows: tuple[int, ...] = (3, 6, 12, 24, 48, 168),
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end Moving-Average baseline pipeline with train/test split.
    Structure mirrors sarima_pipeline() / ets_pipeline().

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  MOVING-AVERAGE PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    # ── Step 1 — window selection (train only) ──────────
    print(f"\n[Window Search] Candidate windows: {sorted(set(candidate_windows))}")
    best_params = select_window(series_train, list(candidate_windows))
    w = best_params["window"]

    # ── Step 2 — forecast for len(test) steps ───────────
    forecast_df = forecast_moving_average(series_train, series_test, window=w)

    # ── Step 3 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df,
                 model_name="moving_average", postfix=postfix)

    # ── Step 4 — metrics ────────────────────────────────
    metrics = evaluate(series_test, forecast_df)

    # ── Print side-by-side comparison ───────────────────
    print("\n[Forecast vs. True]")
    comparison = forecast_df.copy()
    comparison.insert(0, "true", series_test.values)
    print(comparison.to_string())

    return forecast_df, best_params, metrics


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

def get_example_ts() -> pd.Series:
    np.random.seed(42)
    n, period = 300, 12
    t = np.arange(n)
    signal = (
        10 * np.sin(2 * np.pi * t / period)
        + 0.05 * t
        + np.random.normal(0, 1.5, n)
    )
    index = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.Series(signal, index=index, name="value")


def main_moving_average(ts, postfix):
    forecast_df, best_params, metrics = moving_average_pipeline(
        ts,
        max_period=50,
        candidate_windows=(3, 6, 12, 24, 48, 168),
        train_ratio=0.8,
        postfix=postfix,
    )
    return forecast_df, best_params, metrics


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = moving_average_pipeline(
        df_hub1,
        max_period=50,
        candidate_windows=(3, 6, 12, 24, 48, 168),
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = moving_average_pipeline(
        df_hub2,
        max_period=50,
        candidate_windows=(3, 6, 12, 24, 48, 168),
        train_ratio=0.8,
        postfix="hub2",
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.sort_index(inplace=True)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    out_path = Path(__file__).resolve().parent.parent / "data" / "forecast_moving_average.csv"
    forecast_df.to_csv(out_path)

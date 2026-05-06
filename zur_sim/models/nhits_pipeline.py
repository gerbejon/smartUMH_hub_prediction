"""
N-HiTS Forecasting Pipeline
=============================
Reuses shared utilities from prophet_pipeline.py:
  - detect_periodicity()  (FFT + ACF)
  - plot_results()        (train | true test | forecast + CI)
  - evaluate()            (MAE, RMSE, MAPE)
  - get_data()

N-HiTS-specific replacements (steps 3-5):
  - grid_search_nhits()  -> tunes n_blocks, mlp_units, n_pool_kernel_size
  - fit_nhits()          -> fits NHiTS via neuralforecast
  - forecast_nhits()     -> returns forecast_df in the shared format
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical to prophet_pipeline.py)
# ─────────────────────────────────────────────

def detect_periodicity(series: pd.Series, top_n: int = 3, max_period: int = 365) -> list[int]:
    values = series.dropna().values
    n = len(values)

    fft_vals = np.abs(np.fft.rfft(values - values.mean()))
    freqs    = np.fft.rfftfreq(n)

    valid = (freqs > 0) & (1 / freqs <= max_period) & (1 / freqs >= 2)
    fft_vals[~valid] = 0

    peak_indices, _ = find_peaks(fft_vals, height=np.percentile(fft_vals[fft_vals > 0], 75))
    if len(peak_indices) == 0:
        peak_indices = np.argsort(fft_vals)[-top_n:]

    candidate_periods = sorted(
        set(int(round(1 / freqs[i])) for i in peak_indices if freqs[i] > 0),
        key=lambda p: -fft_vals[np.argmin(np.abs(1 / freqs[freqs > 0] - p))]
    )

    nlags = min(max_period + 10, n // 2 - 1)
    acf_vals = acf(values, nlags=nlags, fft=True)

    confirmed = []
    for p in candidate_periods:
        if p < len(acf_vals) and acf_vals[p] > 0.1:
            confirmed.append(p)

    result = (confirmed or candidate_periods)[:top_n]
    print(f"[Periodicity] Detected periods (best first): {result}")
    return result


def plot_results(
    series_train: pd.Series,
    series_test: pd.Series,
    forecast_df: pd.DataFrame,
    postfix: str = "",
    title: str = "N-HiTS — Train / Test Forecast",
):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(series_train.index, series_train.values,
            label="Train (observed)", color="#2563eb", linewidth=1.5)
    ax.plot(series_test.index, series_test.values,
            label="Test (true)", color="#16a34a", linewidth=1.5)
    ax.plot(forecast_df.index, forecast_df["forecast"],
            label="Forecast", color="#dc2626", linewidth=2, linestyle="--")
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
    fname = f"nhits_forecast_{postfix}.png" if postfix else "nhits_forecast.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[Plot] Saved as {fname}")


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
    df = pd.read_csv("../data/hub_distribution.csv", index_col=0)
    df_hub1 = df.iloc[:, 0].fillna(0)
    df_hub2 = df.iloc[:, 1].fillna(0)
    return df_hub1, df_hub2


# ─────────────────────────────────────────────
# N-HiTS HELPER: series -> NeuralForecast DataFrame
# ─────────────────────────────────────────────

def to_nhits_df(series: pd.Series, unique_id: str = "series_1") -> pd.DataFrame:
    """
    NeuralForecast requires a long-format DataFrame with columns:
      - unique_id : series identifier (str)
      - ds        : datestamp (DatetimeIndex or integer converted to dates)
      - y         : values

    Unlike Prophet, NeuralForecast handles the timeline internally,
    so we only need a consistent ds — no need to pre-split the index.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        ds = series.index
    else:
        ds = pd.date_range("2000-01-01", periods=len(series), freq="D")

    return pd.DataFrame({
        "unique_id": unique_id,
        "ds"       : ds,
        "y"        : series.values,
    })


# ─────────────────────────────────────────────
# 3.  GRID SEARCH  (N-HiTS-specific)
# ─────────────────────────────────────────────

def grid_search_nhits(
    series_train: pd.Series,
    horizon: int,
    input_size_multiplier: int = 2,
    n_blocks_options: list[list[int]] = [[1, 1, 1], [2, 2, 2]],
    mlp_units_options: list[int] = [128, 256],
    max_steps_options: list[int] = [100, 300],
    verbose: bool = True,
) -> dict:
    """
    Grid-search N-HiTS hyperparameters using RMSE on an inner holdout
    (last 20% of the training set).

    Parameters
    ----------
    series_train          : training portion of the series
    horizon               : forecast horizon (= len(series_test))
    input_size_multiplier : input_size = multiplier * horizon
    n_blocks_options      : number of blocks per stack to try
    mlp_units_options     : hidden units per MLP layer to try
    max_steps_options     : training steps to try

    Returns
    -------
    best_params dict with keys: n_blocks, mlp_units, max_steps
    """
    inner_split = round(0.8 * len(series_train))
    inner_train = series_train.iloc[:inner_split]
    inner_val   = series_train.iloc[inner_split:]

    # NeuralForecast needs at least input_size + horizon rows
    input_size = input_size_multiplier * horizon

    param_grid = list(itertools.product(n_blocks_options, mlp_units_options, max_steps_options))
    total = len(param_grid)
    print(f"\n[Grid Search] Testing {total} N-HiTS combinations ...")

    best_rmse   = np.inf
    best_params = None
    results     = []

    train_df = to_nhits_df(inner_train)

    for i, (n_blocks, mlp_units, max_steps) in enumerate(param_grid, 1):
        try:
            model = NHITS(
                h=len(inner_val),
                input_size=input_size,
                n_blocks=n_blocks,
                mlp_units=[[mlp_units, mlp_units]] * len(n_blocks),
                max_steps=max_steps,
                loss=MSE(),
                val_check_steps=50,
                early_stop_patience_steps=-1,   # disable early stopping in grid search
            )
            nf = NeuralForecast(models=[model], freq=_infer_freq(series_train))
            nf.fit(train_df)

            # Predict on val by appending val ds to the future frame
            val_df   = to_nhits_df(inner_train)   # full inner train as context
            pred_raw = nf.predict(val_df)
            pred = pred_raw["NHITS"].values[:len(inner_val)]
            true = inner_val.values[:len(pred)]

            rmse = np.sqrt(np.mean((true - pred) ** 2))
            results.append({
                "n_blocks": str(n_blocks), "mlp_units": mlp_units,
                "max_steps": max_steps, "rmse": rmse,
            })

            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = {"n_blocks": n_blocks, "mlp_units": mlp_units, "max_steps": max_steps}

            if verbose:
                print(f"  [{i}/{total}] n_blocks={n_blocks}, mlp_units={mlp_units}, "
                      f"max_steps={max_steps} -> RMSE={rmse:.4f}")

        except Exception as e:
            print(f"  [{i}/{total}] FAILED: {e}")

    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(f"\n[Grid Search] All results:")
    print(df_results.to_string(index=False))
    print(f"\n[Best Model] {best_params}  RMSE={best_rmse:.4f}")
    return best_params


# ─────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────

def _infer_freq(series: pd.Series) -> str:
    """Return pandas freq string from series index, defaulting to 'D'."""
    if isinstance(series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(series.index)
        return freq if freq else "D"
    return "D"


# ─────────────────────────────────────────────
# 4.  FIT  (N-HiTS-specific)
# ─────────────────────────────────────────────

def fit_nhits(
    series_train: pd.Series,
    params: dict,
    horizon: int,
    input_size_multiplier: int = 2,
) -> NeuralForecast:
    """
    Fit N-HiTS on the full training set using the best hyperparameters.

    Returns the fitted NeuralForecast object (which wraps the model).
    """
    input_size = input_size_multiplier * horizon
    n_blocks   = params["n_blocks"]

    model = NHITS(
        h=horizon,
        input_size=input_size,
        n_blocks=n_blocks,
        mlp_units=[[params["mlp_units"], params["mlp_units"]]] * len(n_blocks),
        max_steps=params["max_steps"],
        loss=MSE(),
    )

    nf = NeuralForecast(models=[model], freq=_infer_freq(series_train))
    train_df = to_nhits_df(series_train)
    nf.fit(train_df)

    print(f"\n[N-HiTS] Fitted with horizon={horizon}, input_size={input_size}, params={params}")
    return nf


# ─────────────────────────────────────────────
# 5.  FORECAST  (N-HiTS-specific)
# ─────────────────────────────────────────────

def forecast_nhits(
    nf: NeuralForecast,
    series_train: pd.Series,
    series_test: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Forecast for the test period and return a DataFrame in the shared format:
    columns = [forecast, lower_ci, upper_ci], index = series_test.index

    N-HiTS doesn't produce native CIs, so we estimate them from a simple
    bootstrap of the in-sample residuals (scaled by the normal quantile).
    """
    train_df = to_nhits_df(series_train)
    raw      = nf.predict(train_df)
    forecast = raw["NHITS"].values[:len(series_test)]

    # Estimate CI width from in-sample residuals
    in_sample   = nf.predict_insample(step_size=1)
    residuals   = in_sample["y"].values - in_sample["NHITS"].values
    std         = np.nanstd(residuals)
    from scipy.stats import norm
    z           = norm.ppf(1 - (1 - confidence) / 2)
    margin      = z * std

    return pd.DataFrame({
        "forecast" : forecast,
        "lower_ci" : forecast - margin,
        "upper_ci" : forecast + margin,
    }, index=series_test.index[:len(forecast)])


# ─────────────────────────────────────────────
# 8.  FULL PIPELINE
# ─────────────────────────────────────────────

def nhits_pipeline(
    series: pd.Series,
    max_period: int = 365,
    n_blocks_options: list[list[int]] = [[1, 1, 1], [2, 2, 2]],
    mlp_units_options: list[int] = [128, 256],
    max_steps_options: list[int] = [100, 300],
    input_size_multiplier: int = 2,
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end N-HiTS pipeline with train/test split.
    Structure mirrors sarima_pipeline() and prophet_pipeline() exactly.

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  N-HiTS PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    horizon      = len(series_test)
    print(f"[Split] Train: {len(series_train)} obs | Test: {horizon} obs | Horizon: {horizon}")

    # ── Step 1 — periodicity (informational for N-HiTS) ─
    # N-HiTS learns seasonality implicitly via its multi-rate
    # pooling, but we still detect and print periods for reference.
    detect_periodicity(series_train, max_period=max_period)

    # ── Step 2 — grid search (train only) ───────────────
    best_params = grid_search_nhits(
        series_train,
        horizon=horizon,
        input_size_multiplier=input_size_multiplier,
        n_blocks_options=n_blocks_options,
        mlp_units_options=mlp_units_options,
        max_steps_options=max_steps_options,
    )

    # ── Step 3 — fit (train only) ───────────────────────
    nf = fit_nhits(series_train, best_params, horizon, input_size_multiplier)

    # ── Step 4 — forecast for len(test) steps ───────────
    forecast_df = forecast_nhits(nf, series_train, series_test)

    # ── Step 5 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df, postfix=postfix)

    # ── Step 6 — metrics ────────────────────────────────
    metrics = evaluate(series_test, forecast_df)

    # ── Print side-by-side comparison ───────────────────
    print("\n[Forecast vs. True]")
    comparison = forecast_df.copy()
    comparison.insert(0, "true", series_test.values[:len(forecast_df)])
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


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = nhits_pipeline(
        df_hub1,
        max_period=50,
        n_blocks_options=[[1, 1, 1], [2, 2, 2]],
        mlp_units_options=[128, 256],
        max_steps_options=[100, 300],
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = nhits_pipeline(
        df_hub2,
        max_period=50,
        n_blocks_options=[[1, 1, 1], [2, 2, 2]],
        mlp_units_options=[128, 256],
        max_steps_options=[100, 300],
        train_ratio=0.8,
        postfix="hub2",
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.sort_index(inplace=True)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    forecast_df.to_csv('../data/forecast_nhits.csv')
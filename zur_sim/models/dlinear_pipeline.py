"""
DLinear Forecasting Pipeline
=============================
DLinear (Decomposition-Linear) is a single-layer linear model that splits the
series into trend + seasonal components and forecasts each with a linear layer.
Despite its simplicity it is a strong baseline that often matches or beats heavy
transformers on short series — a useful sanity check against TFT / N-HiTS here.

Mirrors nhits_pipeline.py so it plugs into calculate_all.py unchanged:
  - grid_search_dlinear()  -> tunes input_size_multiplier, moving_avg_window, max_steps
  - fit_dlinear()          -> fits DLinear via neuralforecast
  - forecast_dlinear()     -> returns forecast_df in the shared format

Reuses the shared layout from the other pipelines:
  - plot_results()  (train | true test | forecast + CI) from plot_utils
  - evaluate()      (MAE, RMSE, MAPE)
  - get_data()      (../data/hub_distribution.csv, two hub columns)
"""

import warnings
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from neuralforecast.losses.pytorch import MSE

from plot_utils import plot_results

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical layout to nhits_pipeline.py)
# ─────────────────────────────────────────────

def detect_periodicity(series: pd.Series, top_n: int = 3, max_period: int = 365) -> list[int]:
    """
    Detect the dominant seasonal periods of a series via FFT, confirmed by ACF.

    Peaks in the FFT spectrum (restricted to periods in [2, max_period]) give
    candidate periods; those with autocorrelation > 0.1 at their lag are kept.

    Returns the top_n periods, strongest first (informational only here).
    """
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


def evaluate(series_test: pd.Series, forecast_df: pd.DataFrame) -> dict:
    """
    Compute MAE, RMSE and MAPE between the true test values and the forecast,
    print them, and return them as a dict {"mae", "rmse", "mape"}.
    """
    true = series_test.values[:len(forecast_df)]
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
    """
    Load ../data/hub_distribution.csv and return the two hub share series
    (df_hub1, df_hub2) as NaN-filled (0) pandas Series.
    """
    data_path = Path(__file__).resolve().parent.parent / "data" / "hub_distribution.csv"
    df = pd.read_csv(data_path, index_col=0)
    df_hub1 = df.iloc[:, 0].fillna(0)
    df_hub2 = df.iloc[:, 1].fillna(0)
    return df_hub1, df_hub2


def _infer_freq(series: pd.Series) -> str:
    """Return pandas freq string from series index, defaulting to 'D'."""
    if isinstance(series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(series.index)
        return freq if freq else "D"
    return "D"


# ─────────────────────────────────────────────
# HELPER: series -> NeuralForecast DataFrame
# ─────────────────────────────────────────────

def to_nf_df(series: pd.Series, unique_id: str = "series_1") -> pd.DataFrame:
    """
    NeuralForecast requires a long-format DataFrame with columns:
      - unique_id : series identifier (str)
      - ds        : datestamp
      - y         : values
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
# 3.  GRID SEARCH  (DLinear-specific)
# ─────────────────────────────────────────────

def grid_search_dlinear(
    series_train: pd.Series,
    horizon: int,
    input_size_multiplier_options: list[int] = [2, 3],
    moving_avg_window_options: list[int] = [25, 49],
    max_steps_options: list[int] = [200, 500],
    verbose: bool = True,
) -> dict:
    """
    Grid-search DLinear hyperparameters using RMSE on an inner holdout
    (last 20% of the training set).

    Returns
    -------
    best_params dict with keys: input_size_multiplier, moving_avg_window, max_steps
    """
    inner_split = round(0.8 * len(series_train))
    inner_train = series_train.iloc[:inner_split]
    inner_val   = series_train.iloc[inner_split:]

    param_grid = list(itertools.product(
        input_size_multiplier_options, moving_avg_window_options, max_steps_options
    ))
    total = len(param_grid)
    print(f"\n[Grid Search] Testing {total} DLinear combinations ...")

    best_rmse   = np.inf
    best_params = None
    results     = []

    train_df = to_nf_df(inner_train)

    for i, (ism, maw, max_steps) in enumerate(param_grid, 1):
        try:
            model = DLinear(
                h=len(inner_val),
                input_size=ism * horizon,
                moving_avg_window=maw,
                max_steps=max_steps,
                loss=MSE(),
                val_check_steps=50,
                early_stop_patience_steps=-1,   # disable early stopping in grid search
            )
            nf = NeuralForecast(models=[model], freq=_infer_freq(series_train))
            nf.fit(train_df)

            pred_raw = nf.predict(train_df)
            pred = pred_raw["DLinear"].values[:len(inner_val)]
            true = inner_val.values[:len(pred)]

            rmse = np.sqrt(np.mean((true - pred) ** 2))
            results.append({
                "input_size_multiplier": ism, "moving_avg_window": maw,
                "max_steps": max_steps, "rmse": rmse,
            })

            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = {"input_size_multiplier": ism,
                               "moving_avg_window": maw, "max_steps": max_steps}

            if verbose:
                print(f"  [{i}/{total}] input_size_multiplier={ism}, "
                      f"moving_avg_window={maw}, max_steps={max_steps} -> RMSE={rmse:.4f}")

        except Exception as e:
            print(f"  [{i}/{total}] FAILED: {e}")

    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(f"\n[Grid Search] All results:")
    print(df_results.to_string(index=False))
    print(f"\n[Best Model] {best_params}  RMSE={best_rmse:.4f}")
    return best_params


# ─────────────────────────────────────────────
# 4.  FIT  (DLinear-specific)
# ─────────────────────────────────────────────

def fit_dlinear(
    series_train: pd.Series,
    params: dict,
    horizon: int,
) -> NeuralForecast:
    """
    Fit DLinear on the full training set using the best hyperparameters.
    Returns the fitted NeuralForecast object.
    """
    input_size = params["input_size_multiplier"] * horizon

    model = DLinear(
        h=horizon,
        input_size=input_size,
        moving_avg_window=params["moving_avg_window"],
        max_steps=params["max_steps"],
        loss=MSE(),
    )

    nf = NeuralForecast(models=[model], freq=_infer_freq(series_train))
    train_df = to_nf_df(series_train)
    nf.fit(train_df)

    print(f"\n[DLinear] Fitted with horizon={horizon}, input_size={input_size}, params={params}")
    return nf


# ─────────────────────────────────────────────
# 5.  FORECAST  (DLinear-specific)
# ─────────────────────────────────────────────

def forecast_dlinear(
    nf: NeuralForecast,
    series_train: pd.Series,
    series_test: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Forecast for the test period and return a DataFrame in the shared format:
    columns = [forecast, lower_ci, upper_ci], index = series_test.index

    DLinear doesn't produce native CIs, so we estimate them from the std of the
    in-sample residuals (scaled by the normal quantile).
    """
    train_df = to_nf_df(series_train)
    raw      = nf.predict(train_df)
    forecast = raw["DLinear"].values[:len(series_test)]

    in_sample   = nf.predict_insample(step_size=1)
    residuals   = in_sample["y"].values - in_sample["DLinear"].values
    std         = np.nanstd(residuals)
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

def dlinear_pipeline(
    series: pd.Series,
    max_period: int = 365,
    input_size_multiplier_options: list[int] = [2, 3],
    moving_avg_window_options: list[int] = [25, 49],
    max_steps_options: list[int] = [200, 500],
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end DLinear pipeline with train/test split.
    Structure mirrors nhits_pipeline().

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  DLINEAR PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    horizon      = len(series_test)
    print(f"[Split] Train: {len(series_train)} obs | Test: {horizon} obs | Horizon: {horizon}")

    # ── Step 1 — periodicity (informational) ────────────
    detect_periodicity(series_train, max_period=max_period)

    # ── Step 2 — grid search (train only) ───────────────
    best_params = grid_search_dlinear(
        series_train,
        horizon=horizon,
        input_size_multiplier_options=input_size_multiplier_options,
        moving_avg_window_options=moving_avg_window_options,
        max_steps_options=max_steps_options,
    )

    # ── Step 3 — fit (train only) ───────────────────────
    nf = fit_dlinear(series_train, best_params, horizon)

    # ── Step 4 — forecast for len(test) steps ───────────
    forecast_df = forecast_dlinear(nf, series_train, series_test)

    # ── Step 5 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df,
                 model_name="dlinear", postfix=postfix)

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
    """
    Build a synthetic hourly series (sine seasonality + linear trend + noise)
    for quick demos/testing of the pipeline. Returns a datetime-indexed Series.
    """
    np.random.seed(42)
    n, period = 24 * 30, 24
    t = np.arange(n)
    signal = (
        10 * np.sin(2 * np.pi * t / period)
        + 0.01 * t
        + np.random.normal(0, 1.5, n)
    )
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(signal, index=index, name="value")


def main_dlinear(ts, postfix):
    """
    Convenience entry point: run dlinear_pipeline() on `ts` with default option
    grids and `postfix` for output naming. Returns (forecast_df, best_params, metrics).
    """
    forecast_df, best_params, metrics = dlinear_pipeline(
        ts,
        max_period=50,
        input_size_multiplier_options=[2, 3],
        moving_avg_window_options=[25, 49],
        max_steps_options=[200, 500],
        train_ratio=0.8,
        postfix=postfix,
    )
    return forecast_df, best_params, metrics


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = dlinear_pipeline(
        df_hub1,
        max_period=50,
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = dlinear_pipeline(
        df_hub2,
        max_period=50,
        train_ratio=0.8,
        postfix="hub2",
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.sort_index(inplace=True)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    out_path = Path(__file__).resolve().parent.parent / "data" / "forecast_dlinear.csv"
    forecast_df.to_csv(out_path)

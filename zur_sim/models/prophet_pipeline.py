"""
Prophet Forecasting Pipeline
=============================
Reuses shared utilities from sarima_pipeline.py:
  - detect_periodicity()  (FFT + ACF)
  - plot_results()        (train | true test | forecast + CI)
  - evaluate()            (MAE, RMSE, MAPE)
  - get_data()

Prophet-specific replacements (steps 3-5):
  - grid_search_prophet()  -> tunes changepoint_prior_scale & seasonality_prior_scale
  - fit_prophet()          -> fits Prophet with detected custom seasonality
  - forecast_prophet()     -> returns forecast_df in the shared format
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf, adfuller
from prophet import Prophet

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical to sarima_pipeline.py)
# ─────────────────────────────────────────────

def detect_periodicity(series: pd.Series, top_n: int = 3, max_period: int = 365) -> list[int]:
    """
    Detect the dominant seasonal period(s) of a series via FFT + ACF.

    Peaks of the FFT power spectrum (restricted to periods in [2, max_period])
    give candidate periods, which are then confirmed when their ACF value
    exceeds 0.1. Returns up to top_n periods, strongest first.
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


def plot_results(
    series_train: pd.Series,
    series_test: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Prophet — Train / Test Forecast",
    postfix = ''
):
    """
    Plot training history, true test values, and the forecast with its 95% CI,
    marking the train/test split. Saves the figure to
    prophet_forecast_<postfix>.png.
    """
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
    plt.savefig(f"prophet_forecast_{postfix}.png", dpi=150)
    plt.show()
    print("[Plot] Saved as prophet_forecast.png")


def evaluate(series_test: pd.Series, forecast_df: pd.DataFrame) -> dict:
    """
    Compute and print forecast error metrics (MAE, RMSE, MAPE) comparing the
    true test values against forecast_df['forecast']. Zero-valued truths are
    excluded from MAPE. Returns a dict with keys mae, rmse, mape.
    """
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
    """
    Load data/hub_distribution.csv relative to this module and return the two
    hub share columns as Series with NaNs filled to 0.
    """
    data_path = Path(__file__).resolve().parent.parent / "data" / "hub_distribution.csv"
    df = pd.read_csv(data_path, index_col=0)
    df_hub1 = df.iloc[:, 0].fillna(0)
    df_hub2 = df.iloc[:, 1].fillna(0)
    return df_hub1, df_hub2


# ─────────────────────────────────────────────
# PROPHET HELPER: series -> Prophet DataFrame
# ─────────────────────────────────────────────

def build_ds_index(series: pd.Series) -> pd.DatetimeIndex:
    """
    Build a single consistent DatetimeIndex for the FULL series.
    Call this once on the full series, then slice — never call
    independently on train/test subsets, or the ds values will overlap.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series.index
    else:
        # Integer index: synthesise dates from a fixed origin for the full series
        return pd.date_range("2000-01-01", periods=len(series), freq="D")


def to_prophet_df(series: pd.Series, ds_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Prophet requires a DataFrame with columns 'ds' (datestamp) and 'y' (value).
    Always pass ds_index built from the FULL series so train and test
    share a continuous, non-overlapping timeline.
    """
    return pd.DataFrame({"ds": ds_index, "y": series.values})


# ─────────────────────────────────────────────
# 3.  GRID SEARCH  (Prophet-specific)
# ─────────────────────────────────────────────

def grid_search_prophet(
    series_train: pd.Series,
    periods: list[int],
    changepoint_prior_scales: list[float] = [0.01, 0.1, 0.5],
    seasonality_prior_scales: list[float] = [1.0, 10.0],
    verbose: bool = True,
) -> dict:
    """
    Grid-search Prophet hyperparameters using cross-validation RMSE on a
    simple rolling holdout (last 20% of the training set).

    Parameters
    ----------
    series_train            : training portion of the series
    periods                 : detected seasonal periods (from detect_periodicity)
    changepoint_prior_scale : controls trend flexibility (higher = more flexible)
    seasonality_prior_scale : controls seasonality strength

    Returns
    -------
    best_params dict with keys: changepoint_prior_scale, seasonality_prior_scale
    """
    # Inner holdout: last 20% of train for validation
    inner_split = round(0.8 * len(series_train))
    inner_train = series_train.iloc[:inner_split]
    inner_val   = series_train.iloc[inner_split:]

    # Build ds from the FULL train series so inner_train and inner_val
    # share a continuous, non-overlapping timeline
    ds_train = build_ds_index(series_train)
    train_df = to_prophet_df(inner_train, ds_train[:inner_split])
    val_df   = to_prophet_df(inner_val,   ds_train[inner_split:])

    param_grid = list(itertools.product(changepoint_prior_scales, seasonality_prior_scales))
    total = len(param_grid)
    print(f"\n[Grid Search] Testing {total} Prophet hyperparameter combinations ...")

    best_rmse   = np.inf
    best_params = None
    results     = []

    for i, (cps, sps) in enumerate(param_grid, 1):
        try:
            m = Prophet(
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            # Add each detected period as a custom seasonality
            for p in periods:
                m.add_seasonality(name=f"custom_{p}", period=p, fourier_order=5)

            m.fit(train_df)

            forecast = m.predict(val_df[["ds"]])
            pred = forecast["yhat"].values
            true = inner_val.values

            rmse = np.sqrt(np.mean((true - pred) ** 2))
            results.append({"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "rmse": rmse})

            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}

            if verbose:
                print(f"  [{i}/{total}] cps={cps}, sps={sps} -> RMSE={rmse:.4f}")

        except Exception as e:
            print(f"  [{i}/{total}] cps={cps}, sps={sps} -> FAILED: {e}")

    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(f"\n[Grid Search] All results:")
    print(df_results.to_string(index=False))
    print(f"\n[Best Model] changepoint_prior_scale={best_params['changepoint_prior_scale']}, "
          f"seasonality_prior_scale={best_params['seasonality_prior_scale']}  RMSE={best_rmse:.4f}")
    return best_params


# ─────────────────────────────────────────────
# 4.  FIT  (Prophet-specific)
# ─────────────────────────────────────────────

def fit_prophet(series_train: pd.Series, params: dict, periods: list[int],
                ds_index: pd.DatetimeIndex) -> Prophet:
    """
    Fit Prophet on the full training set using the best hyperparameters.
    Each detected seasonal period is added as a custom seasonality.
    ds_index must be the slice of the full-series ds that covers series_train.
    """
    m = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,   # 95% CI, matching SARIMA pipeline
    )
    for p in periods:
        m.add_seasonality(name=f"custom_{p}", period=p, fourier_order=5)

    train_df = to_prophet_df(series_train, ds_index)
    m.fit(train_df)
    print(f"\n[Prophet] Fitted with periods={periods}, params={params}")
    return m


# ─────────────────────────────────────────────
# 5.  FORECAST  (Prophet-specific)
# ─────────────────────────────────────────────

def forecast_prophet(model: Prophet, series_test: pd.Series,
                     ds_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forecast for the test period and return a DataFrame in the shared format:
    columns = [forecast, lower_ci, upper_ci], index = series_test.index
    ds_index must be the slice of the full-series ds that covers series_test,
    so it continues directly after the training ds values.
    """
    future_df = to_prophet_df(series_test, ds_index)[["ds"]]
    raw = model.predict(future_df)

    return pd.DataFrame({
        "forecast" : raw["yhat"].values,
        "lower_ci" : raw["yhat_lower"].values,
        "upper_ci" : raw["yhat_upper"].values,
    }, index=series_test.index)


# ─────────────────────────────────────────────
# 8.  FULL PIPELINE
# ─────────────────────────────────────────────

def prophet_pipeline(
    series: pd.Series,
    max_period: int = 365,
    changepoint_prior_scales: list[float] = [0.01, 0.1, 0.5],
    seasonality_prior_scales: list[float] = [1.0, 10.0],
    train_ratio: float = 0.8,
    postfix = None
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end Prophet pipeline with train/test split.
    Structure mirrors sarima_pipeline() exactly.

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  PROPHET PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    # ── Build a single ds index for the FULL series ─────
    # Slicing this ensures train and test ds values form a
    # continuous, non-overlapping timeline — critical for
    # correct Prophet forecasting.
    ds_full  = build_ds_index(series)
    ds_train = ds_full[:split_idx]
    ds_test  = ds_full[split_idx:]

    # ── Step 1 — periodicity (train only) ───────────────
    periods = detect_periodicity(series_train, max_period=max_period)

    # ── Step 2 — grid search (train only) ───────────────
    # Note: Prophet handles non-stationarity internally via trend,
    # so select_d / select_D are not needed.
    best_params = grid_search_prophet(
        series_train,
        periods=periods,
        changepoint_prior_scales=changepoint_prior_scales,
        seasonality_prior_scales=seasonality_prior_scales,
    )

    # ── Step 3 — fit (train only) ───────────────────────
    model = fit_prophet(series_train, best_params, periods, ds_train)

    # ── Step 4 — forecast for len(test) steps ───────────
    forecast_df = forecast_prophet(model, series_test, ds_test)

    # ── Step 5 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df, postfix)

    # ── Step 6 — metrics ────────────────────────────────
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
    """
    Build a synthetic monthly demo series (seasonal sine + linear trend +
    Gaussian noise) for quick pipeline testing. Returns a pandas Series with a
    monthly DatetimeIndex.
    """
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

def main_prophet(ts, postfix):
    """
    Convenience entry point: run prophet_pipeline on series ts with the default
    hyperparameter grids and an 80/20 split, tagging outputs with postfix.
    Returns (forecast_df, best_params, metrics).
    """
    forecast_df, best_params, metrics = prophet_pipeline(
        ts,
        max_period=50,
        changepoint_prior_scales=[0.01, 0.1, 0.5],
        seasonality_prior_scales=[1.0, 10.0],
        train_ratio=0.8,
        postfix=postfix
    )
    return forecast_df, best_params, metrics

if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()
    # Hub 1
    postfix = "hub1"
    forecast_df_hub1, best_params_hub1, metrics_hub1 = prophet_pipeline(
        df_hub1,
        max_period=50,
        changepoint_prior_scales=[0.01, 0.1, 0.5],
        seasonality_prior_scales=[1.0, 10.0],
        train_ratio=0.8,

    )
    # Hub 2
    postfix = "hub2"
    forecast_df_hub2, best_params_hub2, metrics_hub2 = prophet_pipeline(
        df_hub2,
        max_period=50,
        changepoint_prior_scales=[0.01, 0.1, 0.5],
        seasonality_prior_scales=[1.0, 10.0],
        train_ratio=0.8,
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.sort_index(inplace=True)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    forecast_df.to_csv('../data/forecast_prophet.csv')
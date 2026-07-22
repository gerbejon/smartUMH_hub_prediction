"""
ETS (Exponential Smoothing) Forecasting Pipeline
==================================================
Reuses shared utilities from sarima_pipeline.py:
  - detect_periodicity()  (FFT + ACF)
  - plot_results()        (train | true test | forecast + CI)
  - evaluate()            (MAE, RMSE, MAPE)
  - get_data()

ETS-specific replacements (steps 3-5):
  - grid_search_ets()  -> tries all valid error/trend/seasonal combinations
  - fit_ets()          -> fits best ETSModel via statsmodels
  - forecast_ets()     -> returns forecast_df in the shared format

Key ETS advantage:
  - No differencing needed (handles non-stationarity via smoothing)
  - statsmodels auto-selects best error/trend/seasonal type via AIC
  - Much faster than SARIMA grid search
"""

import warnings
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

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
    postfix: str = "",
    title: str = "ETS — Train / Test Forecast",
):
    """
    Plot training history, true test values, and the forecast with its 95% CI,
    marking the train/test split. The figure filename is derived from postfix
    (saving is currently commented out; the plot is shown instead).
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
    fname = f"ets_forecast_{postfix}.png" if postfix else "ets_forecast.png"
    # plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[Plot] Saved as {fname}")


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
# 3.  GRID SEARCH  (ETS-specific)
# ─────────────────────────────────────────────

def grid_search_ets(
    series_train: pd.Series,
    s: int,
    verbose: bool = True,
) -> dict:
    """
    Try all valid ETS(error, trend, seasonal) combinations and pick
    the one with the lowest AIC.

    ETS components:
      error    : 'add' (additive) or 'mul' (multiplicative)
      trend    : None, 'add', or 'mul'
      seasonal : None, 'add', or 'mul'
      damped   : True or False (only meaningful when trend is not None)

    Multiplicative components require all-positive data — combinations
    that would fail on non-positive series are skipped automatically.

    Parameters
    ----------
    series_train : training portion of the series
    s            : seasonal period (from detect_periodicity)

    Returns
    -------
    best_params dict with keys: error, trend, seasonal, damped, s
    """
    error_types    = ["add", "mul"]
    trend_types    = [None, "add", "mul"]
    seasonal_types = [None, "add", "mul"]
    damped_options = [False, True]

    all_positive = (series_train > 0).all()

    param_grid = list(itertools.product(
        error_types, trend_types, seasonal_types, damped_options
    ))
    total = len(param_grid)
    print(f"\n[Grid Search] Testing up to {total} ETS combinations (s={s}) ...")

    best_aic    = np.inf
    best_params = None
    results     = []

    for i, (error, trend, seasonal, damped) in enumerate(param_grid, 1):
        # damped only applies when there is a trend
        if damped and trend is None:
            continue
        # multiplicative components require strictly positive data
        if not all_positive and (error == "mul" or trend == "mul" or seasonal == "mul"):
            continue
        # seasonal=None means we don't use the period s
        seasonal_periods = s if seasonal is not None else None

        try:
            model = ETSModel(
                series_train,
                error=error,
                trend=trend,
                seasonal=seasonal,
                damped_trend=damped,
                seasonal_periods=seasonal_periods,
            )
            fit = model.fit(disp=False)
            aic = fit.aic
            results.append({
                "error": error, "trend": trend, "seasonal": seasonal,
                "damped": damped, "aic": aic,
            })

            if aic < best_aic:
                best_aic    = aic
                best_params = {
                    "error"   : error,
                    "trend"   : trend,
                    "seasonal": seasonal,
                    "damped"  : damped,
                    "s"       : seasonal_periods,
                }

            if verbose:
                print(f"  [{i}/{total}] ETS({error},{trend},{seasonal}) "
                      f"damped={damped} -> AIC={aic:.2f}")

        except Exception as e:
            if verbose:
                print(f"  [{i}/{total}] ETS({error},{trend},{seasonal}) "
                      f"damped={damped} -> FAILED: {e}")

    df_results = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    print(f"\n[Grid Search] Top 5 models:")
    print(df_results.head(5).to_string(index=False))
    print(f"\n[Best Model] ETS({best_params['error']},{best_params['trend']},"
          f"{best_params['seasonal']}) damped={best_params['damped']}  AIC={best_aic:.2f}")
    return best_params


# ─────────────────────────────────────────────
# 4.  FIT  (ETS-specific)
# ─────────────────────────────────────────────

def fit_ets(series_train: pd.Series, params: dict):
    """
    Fit ETSModel on the full training set using the best parameters.
    Returns the fitted result object.
    """
    model = ETSModel(
        series_train,
        error=params["error"],
        trend=params["trend"],
        seasonal=params["seasonal"],
        damped_trend=params["damped"],
        seasonal_periods=params["s"],
    )
    fit = model.fit(disp=False)
    print("\n[Model Summary]")
    print(fit.summary())
    return fit


# ─────────────────────────────────────────────
# 5.  FORECAST  (ETS-specific)
# ─────────────────────────────────────────────

def forecast_ets(
    fit,
    series_test: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Forecast for the test period and return a DataFrame in the shared format:
    columns = [forecast, lower_ci, upper_ci], index = series_test.index

    ETSResults uses get_prediction(start, end).summary_frame(alpha).
    start = first out-of-sample index (= length of training set)
    end   = last out-of-sample index  (= start + len(test) - 1)
    """
    n_train = fit.model.nobs
    start   = n_train
    end     = n_train + len(series_test) - 1

    pred = fit.get_prediction(start=start, end=end)
    ci   = pred.summary_frame(alpha=alpha)

    return pd.DataFrame({
        "forecast" : ci["mean"].values,
        "lower_ci" : ci["pi_lower"].values,
        "upper_ci" : ci["pi_upper"].values,
    }, index=series_test.index)


# ─────────────────────────────────────────────
# 8.  FULL PIPELINE
# ─────────────────────────────────────────────

def ets_pipeline(
    series: pd.Series,
    max_period: int = 365,
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end ETS pipeline with train/test split.
    Structure mirrors sarima_pipeline() and all subsequent pipelines.

    Note: ETS has no p/q/P/Q range parameters — the grid search
    exhausts all valid error/trend/seasonal type combinations automatically.

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  ETS PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    # ── Step 1 — periodicity (train only) ───────────────
    periods = detect_periodicity(series_train, max_period=max_period)
    s = periods[0]

    # ── Step 2 — grid search (train only) ───────────────
    # No select_d / select_D needed: ETS handles non-stationarity
    # via its level/trend smoothing equations internally.
    best_params = grid_search_ets(series_train, s=s)

    # ── Step 3 — fit (train only) ───────────────────────
    fit = fit_ets(series_train, best_params)

    # ── Step 4 — forecast for len(test) steps ───────────
    forecast_df = forecast_ets(fit, series_test)

    # ── Step 5 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df, postfix=postfix)

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

def main_ets(ts, postfix):
    """
    Convenience entry point: run ets_pipeline on series ts with an 80/20 split,
    tagging outputs with postfix. Returns (forecast_df, best_params, metrics).
    """
    forecast_df, best_params, metrics = ets_pipeline(
        ts,
        max_period=50,
        train_ratio=0.8,
        postfix=postfix,
    )
    return forecast_df, best_params, metrics


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = ets_pipeline(
        df_hub1,
        max_period=50,
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = ets_pipeline(
        df_hub2,
        max_period=50,
        train_ratio=0.8,
        postfix="hub2",
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    forecast_df.to_csv('../data/forecast_ets.csv')
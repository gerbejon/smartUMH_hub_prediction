"""
Seasonal-Naive Forecasting Pipeline
====================================
Baseline pipeline that mirrors the structure of sarima_pipeline.py /
ets_pipeline.py so it plugs into calculate_all.py unchanged.

Seasonal-naive forecast:
  - the test horizon is filled by repeating the last full season of the
    training set (e.g. the last 24h block for an hourly daily cycle, or the
    last 168h block for a weekly cycle).
  - the season length `s` is chosen automatically: candidate periods come
    from FFT+ACF periodicity detection (shared with the other pipelines)
    plus the canonical hourly seasons {24, 168}; the candidate with the
    lowest in-sample seasonal-naive MAE is selected.
  - confidence band is derived from the std of seasonal training residuals.

Reuses the shared layout from the other pipelines:
  - detect_periodicity()  (FFT + ACF)
  - plot_results()        (train | true test | forecast + CI)
  - evaluate()            (MAE, RMSE, MAPE)
  - get_data()            (../data/hub_distribution.csv, two hub columns)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

from plot_utils import plot_results

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical layout to sarima_pipeline.py)
# ─────────────────────────────────────────────

def detect_periodicity(series: pd.Series, top_n: int = 3, max_period: int = 365) -> list[int]:
    """Detect dominant seasonal periods. Candidate periods come from the peaks
    of the FFT power spectrum (restricted to 2..max_period), then confirmed by
    requiring ACF > 0.1 at that lag. Returns up to `top_n` periods, strongest
    first (falls back to the raw FFT candidates if none are ACF-confirmed)."""
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
    """Compute and print test-set MAE, RMSE and MAPE (zero true values excluded
    from MAPE); return them as a dict."""
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
    """Load ../data/hub_distribution.csv (path resolved relative to this file)
    and return the two hub share columns as NaN-filled (0) Series."""
    # Resolve ../data relative to this file so it works from any working dir.
    data_path = Path(__file__).resolve().parent.parent / "data" / "hub_distribution.csv"
    df = pd.read_csv(data_path, index_col=0)
    df_hub1 = df.iloc[:, 0].fillna(0)
    df_hub2 = df.iloc[:, 1].fillna(0)
    return df_hub1, df_hub2


# ─────────────────────────────────────────────
# SEASON SELECTION  (Seasonal-Naive-specific)
# ─────────────────────────────────────────────

def select_season(series_train: pd.Series, candidate_seasons: list[int], verbose: bool = True) -> dict:
    """
    Pick the season length `s` whose seasonal-naive prediction (y_t = y_{t-s})
    has the lowest in-sample MAE on the training set.
    """
    values = series_train.to_numpy(dtype=float)
    n = len(values)

    best_mae = np.inf
    best_s   = None
    results  = []
    for s in sorted(set(int(c) for c in candidate_seasons if 2 <= int(c) < n)):
        pred = values[:-s]
        true = values[s:]
        mae  = float(np.mean(np.abs(true - pred)))
        results.append({"s": s, "in_sample_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_s   = s
        if verbose:
            print(f"  season s={s:4d} -> in-sample MAE={mae:.4f}")

    if best_s is None:  # degenerate fallback
        best_s, best_mae = max(2, n // 2), float("nan")

    print(f"\n[Best Season] s={best_s}  in-sample MAE={best_mae:.4f}")
    return {"season": best_s, "in_sample_mae": best_mae}


# ─────────────────────────────────────────────
# FORECAST  (Seasonal-Naive-specific)
# ─────────────────────────────────────────────

def forecast_seasonal_naive(
    series_train: pd.Series,
    series_test: pd.Series,
    season: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Repeat the last full season of the training set across the test horizon.
    The 95% band uses the std of in-sample seasonal residuals.
    """
    last_season = series_train.iloc[-season:].to_numpy(dtype=float)
    reps        = int(np.ceil(len(series_test) / season))
    forecast    = np.tile(last_season, reps)[: len(series_test)]

    values    = series_train.to_numpy(dtype=float)
    resid_std = float(np.std(values[season:] - values[:-season])) if len(values) > season else 0.0
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

def seasonal_naive_pipeline(
    series: pd.Series,
    max_period: int = 365,
    extra_seasons: tuple[int, ...] = (24, 168),
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end Seasonal-Naive baseline pipeline with train/test split.
    Structure mirrors sarima_pipeline() / ets_pipeline().

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  SEASONAL-NAIVE PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    # ── Step 1 — periodicity (train only) ───────────────
    periods = detect_periodicity(series_train, max_period=max_period)

    # ── Step 2 — season selection (train only) ──────────
    candidates = list(periods) + list(extra_seasons)
    print(f"\n[Season Search] Candidate seasons: {sorted(set(candidates))}")
    best_params = select_season(series_train, candidates)
    s = best_params["season"]

    # ── Step 3 — forecast for len(test) steps ───────────
    forecast_df = forecast_seasonal_naive(series_train, series_test, season=s)

    # ── Step 4 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df,
                 model_name="seasonal_naive", postfix=postfix)

    # ── Step 5 — metrics ────────────────────────────────
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
    """Return a synthetic monthly demo series (300 points, period 12) with a
    sine cycle, mild linear trend and noise, for standalone testing."""
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


def main_seasonal_naive(ts, postfix):
    """calculate_all.py entry point: run seasonal_naive_pipeline on `ts` and
    return (forecast_df, best_params, metrics)."""
    forecast_df, best_params, metrics = seasonal_naive_pipeline(
        ts,
        max_period=50,
        extra_seasons=(24, 168),
        train_ratio=0.8,
        postfix=postfix,
    )
    return forecast_df, best_params, metrics


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = seasonal_naive_pipeline(
        df_hub1,
        max_period=50,
        extra_seasons=(24, 168),
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = seasonal_naive_pipeline(
        df_hub2,
        max_period=50,
        extra_seasons=(24, 168),
        train_ratio=0.8,
        postfix="hub2",
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.sort_index(inplace=True)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    out_path = Path(__file__).resolve().parent.parent / "data" / "forecast_seasonal_naive.csv"
    forecast_df.to_csv(out_path)

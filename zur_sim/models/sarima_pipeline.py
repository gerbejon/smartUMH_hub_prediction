"""
SARIMA Auto-Tuning & Forecasting Pipeline
==========================================
Steps:
  1. Detect dominant periodicity (FFT + ACF)
  2. Grid-search best (p,d,q)(P,D,Q,s) via AIC
  3. Fit model on first 80% of data
  4. Predict on last 20% and plot vs. true values
"""

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  PERIODICITY DETECTION
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


# ─────────────────────────────────────────────
# 2.  STATIONARITY & DIFFERENCING ORDER
# ─────────────────────────────────────────────

def select_d(series: pd.Series, max_d: int = 2) -> int:
    for d in range(max_d + 1):
        s = series.diff(d).dropna() if d > 0 else series.dropna()
        p_value = adfuller(s)[1]
        if p_value < 0.05:
            print(f"[Stationarity] d={d} (ADF p={p_value:.4f})")
            return d
    print(f"[Stationarity] d={max_d} (series may still be non-stationary)")
    return max_d


def select_D(series: pd.Series, s: int, max_D: int = 1) -> int:
    for D in range(max_D + 1):
        s_diff = series.diff(s * D).dropna() if D > 0 else series.dropna()
        p_value = adfuller(s_diff)[1]
        if p_value < 0.05:
            print(f"[Seasonal Stationarity] D={D} (ADF p={p_value:.4f})")
            return D
    return max_D


# ─────────────────────────────────────────────
# 3.  GRID SEARCH FOR BEST (p,d,q)(P,D,Q,s)
# ─────────────────────────────────────────────

def grid_search_sarima(
    series: pd.Series,
    s: int,
    d: int,
    D: int,
    p_range: range = range(0, 3),
    q_range: range = range(0, 3),
    P_range: range = range(0, 2),
    Q_range: range = range(0, 2),
    verbose: bool = True,
) -> dict:
    best_aic    = np.inf
    best_params = None
    results     = []

    param_grid = list(itertools.product(p_range, q_range, P_range, Q_range))
    total = len(param_grid)
    print(f"\n[Grid Search] Testing {total} SARIMA(p,{d},q)(P,{D},Q,{s}) combinations ...")

    for i, (p, q, P, Q) in enumerate(param_grid, 1):
        try:
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            aic = fit.aic
            results.append({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s, "aic": aic})

            if aic < best_aic:
                best_aic    = aic
                best_params = {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s, "aic": aic}

            if verbose and i % 10 == 0:
                print(f"  {i}/{total} tested | best AIC so far: {best_aic:.2f}")

        except Exception:
            pass

    df_results = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    print(f"\n[Grid Search] Top 5 models:")
    print(df_results.head(5).to_string(index=False))
    print(f"\n[Best Model] SARIMA({best_params['p']},{d},{best_params['q']})"
          f"({best_params['P']},{D},{best_params['Q']},{s})  AIC={best_aic:.2f}")
    return best_params


# ─────────────────────────────────────────────
# 4.  FIT FINAL MODEL
# ─────────────────────────────────────────────

def fit_sarima(series: pd.Series, params: dict):
    model = SARIMAX(
        series,
        order=(params["p"], params["d"], params["q"]),
        seasonal_order=(params["P"], params["D"], params["Q"], params["s"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    print("\n[Model Summary]")
    print(fit.summary())
    return fit


# ─────────────────────────────────────────────
# 5.  FORECAST  (aligned to test index)
# ─────────────────────────────────────────────

def forecast_sarima(fit, series_test: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """
    Forecast exactly len(series_test) steps and align to the test index.
    """
    steps = len(series_test)
    pred  = fit.get_forecast(steps=steps)
    mean  = pred.predicted_mean
    ci    = pred.conf_int(alpha=alpha)

    return pd.DataFrame({
        "forecast" : mean.values,
        "lower_ci" : ci.iloc[:, 0].values,
        "upper_ci" : ci.iloc[:, 1].values,
    }, index=series_test.index)


# ─────────────────────────────────────────────
# 6.  PLOT  (train | true test | forecast + CI)
# ─────────────────────────────────────────────

def plot_results(
    series_train: pd.Series,
    series_test: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "SARIMA — Train / Test Forecast",
    plot_short = False
):
    fig, ax = plt.subplots(figsize=(14, 5))
    if plot_short:
        series_train = series_train.iloc[round(len(series_train) * 0.8):]
    # Training history
    ax.plot(series_train.index, series_train.values,
            label="Train (observed)", color="#2563eb", linewidth=1.5)

    # True test values
    ax.plot(series_test.index, series_test.values,
            label="Test (true)", color="#16a34a", linewidth=1.5)

    # Forecast
    ax.plot(forecast_df.index, forecast_df["forecast"],
            label="Forecast", color="#dc2626", linewidth=2, linestyle="--")

    # Confidence interval
    ax.fill_between(
        forecast_df.index,
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        color="#dc2626", alpha=0.15, label="95% CI",
    )

    # Train/test split line
    ax.axvline(series_test.index[0], color="gray", linestyle=":",
               linewidth=1.5, label="Train/test split")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"sarima_forecast_{postfix}.png", dpi=150)
    plt.show()
    print("[Plot] Saved as sarima_forecast.png")


# ─────────────────────────────────────────────
# 7.  EVALUATION METRICS
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


# ─────────────────────────────────────────────
# 8.  FULL PIPELINE
# ─────────────────────────────────────────────

def sarima_pipeline(
    series: pd.Series,
    max_period: int = 365,
    p_range: range = range(0, 3),
    q_range: range = range(0, 3),
    P_range: range = range(0, 2),
    Q_range: range = range(0, 2),
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end SARIMA pipeline with train/test split.
    All fitting, period detection, and differencing selection
    is done exclusively on the training set.

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  SARIMA AUTO-TUNING PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    # ── Step 1 — periodicity (train only) ───────────────
    periods = detect_periodicity(series_train, max_period=max_period)
    s = periods[0]

    # ── Step 2 — differencing orders (train only) ───────
    d = select_d(series_train)
    D = select_D(series_train, s)

    # ── Step 3 — grid search (train only) ───────────────
    best_params = grid_search_sarima(
        series_train, s=s, d=d, D=D,
        p_range=p_range, q_range=q_range,
        P_range=P_range, Q_range=Q_range,
    )

    # ── Step 4 — fit (train only) ───────────────────────
    fit = fit_sarima(series_train, best_params)

    # ── Step 5 — forecast for len(test) steps ───────────
    forecast_df = forecast_sarima(fit, series_test)

    # ── Step 6 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df, plot_short=True)

    # ── Step 7 — metrics ────────────────────────────────
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


def get_data():
    df = pd.read_csv("../data/hub_distribution.csv", index_col=0)
    df_hub1 = df.iloc[:, 0].fillna(0)
    df_hub2 = df.iloc[:, 1].fillna(0)
    return df_hub1, df_hub2


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()
    # Hub 1
    postfix = "hub1"
    forecast_df_hub1, best_params_hub1, metrics_hub1 = sarima_pipeline(
        df_hub1,
        max_period=50,
        p_range=range(0, 5),
        q_range=range(0, 5),
        P_range=range(0, 2),
        Q_range=range(0, 2),
        train_ratio=0.8,
    )
    # Hub 2
    postfix = "hub2"
    forecast_df_hub2, best_params_hub2, metrics_hub2 = sarima_pipeline(
        df_hub2,
        max_period=50,
        p_range=range(0, 5),
        q_range=range(0, 5),
        P_range=range(0, 2),
        Q_range=range(0, 2),
        train_ratio=0.8,
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.sort_index(inplace=True)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    forecast_df.to_csv('../data/forecast_sarima.csv')
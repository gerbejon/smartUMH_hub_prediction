"""
Temporal Fusion Transformer (TFT) Forecasting Pipeline
========================================================
Reuses shared utilities from nhits_pipeline.py:
  - detect_periodicity()  (FFT + ACF)
  - plot_results()        (train | true test | forecast + CI)
  - evaluate()            (MAE, RMSE, MAPE)
  - get_data()
  - to_nf_df()            (series -> NeuralForecast long-format DataFrame)
  - _infer_freq()

TFT-specific replacements (steps 3-5):
  - grid_search_tft()  -> tunes hidden_size, attention_head_size, dropout
  - fit_tft()          -> fits TFT via neuralforecast
  - forecast_tft()     -> returns forecast_df in the shared format

Key TFT advantage over N-HiTS:
  - Outputs attention weights -> interpretable (which time steps mattered most)
  - plot_attention() visualises these weights after forecasting
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MSE
from scipy.stats import norm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical to nhits_pipeline.py)
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
    title: str = "TFT — Train / Test Forecast",
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
    fname = f"tft_forecast_{postfix}.png" if postfix else "tft_forecast.png"
    # plt.savefig(fname, dpi=150)
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


def _infer_freq(series: pd.Series) -> str:
    """Return pandas freq string from series index, defaulting to 'D'."""
    if isinstance(series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(series.index)
        return freq if freq else "D"
    return "D"


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
# TFT EXTRA: ATTENTION PLOT
# ─────────────────────────────────────────────

def plot_attention(
    nf: NeuralForecast,
    series_train: pd.Series,
    postfix: str = "",
):
    """
    Plot the temporal self-attention weights from the fitted TFT model.
    These show which past time steps the model focused on when forecasting —
    the key interpretability advantage of TFT over N-HiTS and SARIMA.

    Attention is averaged across all heads and forecast steps.
    """
    try:
        train_df   = to_nf_df(series_train)
        raw        = nf.predict(train_df)

        # Retrieve attention weights stored on the underlying model
        tft_model  = nf.models[0]
        attn       = tft_model.decomposition_attention   # shape: (heads, steps, input_size)
        attn_mean  = attn.mean(axis=(0, 1))              # average over heads and forecast steps
        attn_mean  = attn_mean / attn_mean.sum()         # normalise to sum=1

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(attn_mean)), attn_mean, color="#7c3aed", alpha=0.75)
        ax.set_title("TFT Temporal Attention Weights\n"
                     "(higher = model paid more attention to that past step)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Steps back in time (0 = most recent)")
        ax.set_ylabel("Normalised attention weight")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fname = f"tft_attention_{postfix}.png" if postfix else "tft_attention.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"[Attention Plot] Saved as {fname}")

    except AttributeError:
        print("[Attention Plot] Attention weights not available on this version of neuralforecast — skipping.")


# ─────────────────────────────────────────────
# 3.  GRID SEARCH  (TFT-specific)
# ─────────────────────────────────────────────

def grid_search_tft(
    series_train: pd.Series,
    horizon: int,
    input_size_multiplier: int = 2,
    hidden_size_options: list[int] = [32, 64],
    attention_head_size_options: list[int] = [1, 2],
    dropout_options: list[float] = [0.1, 0.3],
    max_steps_options: list[int] = [100, 300],
    verbose: bool = True,
) -> dict:
    """
    Grid-search TFT hyperparameters using RMSE on an inner holdout
    (last 20% of the training set).

    Parameters
    ----------
    hidden_size           : size of hidden layers in the gated residual networks
    attention_head_size   : number of multi-head attention heads
    dropout               : dropout rate (regularisation)
    max_steps             : training steps
    """
    inner_split = round(0.8 * len(series_train))
    inner_train = series_train.iloc[:inner_split]
    inner_val   = series_train.iloc[inner_split:]

    input_size = input_size_multiplier * horizon
    train_df   = to_nf_df(inner_train)

    param_grid = list(itertools.product(
        hidden_size_options,
        attention_head_size_options,
        dropout_options,
        max_steps_options,
    ))
    total = len(param_grid)
    print(f"\n[Grid Search] Testing {total} TFT combinations ...")

    best_rmse   = np.inf
    best_params = None
    results     = []

    for i, (hidden_size, attn_heads, dropout, max_steps) in enumerate(param_grid, 1):
        try:
            model = TFT(
                h=len(inner_val),
                input_size=input_size,
                hidden_size=hidden_size,
                # attention_head_size=attn_heads,
                dropout=dropout,
                max_steps=max_steps,
                loss=MSE(),
                val_check_steps=50,
                early_stop_patience_steps=-1,
                batch_size=16,         # smaller batches
                windows_batch_size=64, # fewer windows loaded at once
            )
            nf = NeuralForecast(models=[model], freq=_infer_freq(series_train))
            nf.fit(train_df)

            pred_raw = nf.predict(train_df)
            pred = pred_raw["TFT"].values[:len(inner_val)]
            true = inner_val.values[:len(pred)]

            rmse = np.sqrt(np.mean((true - pred) ** 2))
            results.append({
                "hidden_size": hidden_size, "attn_heads": attn_heads,
                "dropout": dropout,
                "max_steps": max_steps,
                "rmse": rmse,
            })

            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = {
                    "hidden_size"        : hidden_size,
                    "attention_head_size": attn_heads,
                    "dropout"            : dropout,
                    "max_steps"          : max_steps,
                }

            if verbose:
                print(f"  [{i}/{total}] hidden={hidden_size}, heads={attn_heads}, "
                      f"dropout={dropout}, steps={max_steps} -> RMSE={rmse:.4f}")

        except Exception as e:
            print(f"  [{i}/{total}] FAILED: {e}")

    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(f"\n[Grid Search] All results:")
    print(df_results.to_string(index=False))
    print(f"\n[Best Model] {best_params}  RMSE={best_rmse:.4f}")
    return best_params


# ─────────────────────────────────────────────
# 4.  FIT  (TFT-specific)
# ─────────────────────────────────────────────

def fit_tft(
    series_train: pd.Series,
    params: dict,
    horizon: int,
    input_size_multiplier: int = 2,
) -> NeuralForecast:
    """
    Fit TFT on the full training set using the best hyperparameters.
    Returns the fitted NeuralForecast object.
    """
    input_size = input_size_multiplier * horizon

    model = TFT(
        h=horizon,
        input_size=input_size,
        hidden_size=params["hidden_size"],
        # attention_head_size=params["attention_head_size"],
        dropout=params["dropout"],
        max_steps=params["max_steps"],
        loss=MSE(),
    )

    nf = NeuralForecast(models=[model], freq=_infer_freq(series_train))
    train_df = to_nf_df(series_train)
    nf.fit(train_df)

    print(f"\n[TFT] Fitted with horizon={horizon}, input_size={input_size}, params={params}")
    return nf


# ─────────────────────────────────────────────
# 5.  FORECAST  (TFT-specific)
# ─────────────────────────────────────────────

def forecast_tft(
    nf: NeuralForecast,
    series_train: pd.Series,
    series_test: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Forecast for the test period and return a DataFrame in the shared format:
    columns = [forecast, lower_ci, upper_ci], index = series_test.index

    CI is estimated from in-sample residuals (same approach as N-HiTS pipeline).
    """
    train_df = to_nf_df(series_train)
    raw      = nf.predict(train_df)
    forecast = raw["TFT"].values[:len(series_test)]

    # Estimate CI from in-sample residuals
    in_sample = nf.predict_insample(step_size=1)
    residuals = in_sample["y"].values - in_sample["TFT"].values
    std       = np.nanstd(residuals)
    z         = norm.ppf(1 - (1 - confidence) / 2)
    margin    = z * std

    return pd.DataFrame({
        "forecast" : forecast,
        "lower_ci" : forecast - margin,
        "upper_ci" : forecast + margin,
    }, index=series_test.index[:len(forecast)])


# ─────────────────────────────────────────────
# 8.  FULL PIPELINE
# ─────────────────────────────────────────────

def tft_pipeline(
    series: pd.Series,
    max_period: int = 365,
    hidden_size_options: list[int] = [64, 128],
    attention_head_size_options: list[int] = [1, 2],
    dropout_options: list[float] = [0.1, 0.3],
    max_steps_options: list[int] = [100, 300],
    input_size_multiplier: int = 2,
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end TFT pipeline with train/test split.
    Structure mirrors sarima_pipeline(), prophet_pipeline(), nhits_pipeline().

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  TFT PIPELINE  (80/20 split)")
    print("=" * 55)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    horizon      = len(series_test)
    print(f"[Split] Train: {len(series_train)} obs | Test: {horizon} obs | Horizon: {horizon}")

    # ── Step 1 — periodicity (informational) ────────────
    # TFT learns temporal patterns via self-attention —
    # no manual seasonality specification needed.
    detect_periodicity(series_train, max_period=max_period)

    # ── Step 2 — grid search (train only) ───────────────
    # best_params = grid_search_tft(
    #     series_train,
    #     horizon=horizon,
    #     input_size_multiplier=input_size_multiplier,
    #     hidden_size_options=hidden_size_options,
    #     attention_head_size_options=attention_head_size_options,
    #     dropout_options=dropout_options,
    #     max_steps_options=max_steps_options,
    # )
    best_params = {'hidden_size': 128, 'attention_head_size': 1, 'dropout': 0.3, 'max_steps': 300}

    # ── Step 3 — fit (train only) ───────────────────────
    nf = fit_tft(series_train, best_params, horizon, input_size_multiplier)

    # ── Step 4 — forecast for len(test) steps ───────────
    forecast_df = forecast_tft(nf, series_train, series_test)

    # ── Step 5 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df, postfix=postfix)

    # ── Step 6 — attention plot (TFT-exclusive) ──────────
    plot_attention(nf, series_train, postfix=postfix)

    # ── Step 7 — metrics ────────────────────────────────
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


def main_tft(ts, postfix):
    forecast_df, best_params, metrics = tft_pipeline(
        ts,
        max_period=50,
        hidden_size_options=[64, 128],
        attention_head_size_options=[1, 2],
        dropout_options=[0.1, 0.3],
        max_steps_options=[100, 300],
        train_ratio=0.8,
        postfix=postfix,
    )
    return forecast_df, best_params, metrics

if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = tft_pipeline(
        df_hub1,
        max_period=50,
        hidden_size_options=[64, 128],
        attention_head_size_options=[1, 2],
        dropout_options=[0.1, 0.3],
        max_steps_options=[100, 300],
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = tft_pipeline(
        df_hub2,
        max_period=50,
        hidden_size_options=[64, 128],
        attention_head_size_options=[1, 2],
        dropout_options=[0.1, 0.3],
        max_steps_options=[100, 300],
        train_ratio=0.8,
        postfix="hub2",
    )

    forecast_df = pd.concat([
        forecast_df_hub1.forecast, df_hub1, forecast_df_hub2.forecast, df_hub2
    ], axis=1)
    forecast_df.columns = ['hub1_y_hat', 'hub1_y', 'hub2_y_hat', 'hub2_y']

    forecast_df.to_csv('../data/forecast_tft.csv')
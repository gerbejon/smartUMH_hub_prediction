"""
Flat-Mean Forecasting Pipeline
==============================
The most honest baseline there is: predict every test step as the *constant
training mean*. It captures zero temporal structure, so it is the reference any
"real" model must beat. If a sophisticated model cannot improve on this line,
the series has no exploitable signal at this horizon (see skill_diagnostic.py).

Mirrors naive_pipeline.py so it plugs into calculate_all.py unchanged:
  - main_mean(ts, postfix) -> (forecast_df, best_params, metrics)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from plot_utils import plot_results

warnings.filterwarnings("ignore")


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


def forecast_mean(
    series_train: pd.Series,
    series_test: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Predict every test step as the mean of the training set.
    The 95% band uses the std of the training values around their mean.
    """
    mean_value = float(series_train.mean())
    forecast   = np.full(len(series_test), mean_value, dtype=float)

    resid_std = float(series_train.std())
    z = 1.96  # ~95% normal band (alpha=0.05)
    band = z * resid_std

    return pd.DataFrame({
        "forecast": forecast,
        "lower_ci": forecast - band,
        "upper_ci": forecast + band,
    }, index=series_test.index)


def mean_pipeline(
    series: pd.Series,
    max_period: int = 365,
    train_ratio: float = 0.8,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """End-to-end flat-mean baseline pipeline with train/test split."""
    print("=" * 55)
    print("  MEAN PIPELINE  (80/20 split)")
    print("=" * 55)

    split_idx    = round(train_ratio * len(series))
    series_train = series.iloc[:split_idx]
    series_test  = series.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    best_params = {"strategy": "train_mean", "mean": float(series_train.mean())}

    forecast_df = forecast_mean(series_train, series_test)

    plot_results(series_train, series_test, forecast_df,
                 model_name="mean", postfix=postfix)

    metrics = evaluate(series_test, forecast_df)

    print("\n[Forecast vs. True]")
    comparison = forecast_df.copy()
    comparison.insert(0, "true", series_test.values)
    print(comparison.to_string())

    return forecast_df, best_params, metrics


def main_mean(ts, postfix):
    return mean_pipeline(ts, max_period=50, train_ratio=0.8, postfix=postfix)

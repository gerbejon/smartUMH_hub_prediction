"""
LightGBM (Recursive Multi-Step) Forecasting Pipeline
=====================================================
Gradient-boosted-tree pipeline that mirrors xgboost_pipeline.py so it plugs
into calculate_all.py unchanged. LightGBM is typically faster than XGBoost and
handles many features cheaply; the forecasting setup is otherwise identical:

  - features are autoregressive lags + rolling means + cyclical calendar
    encodings derived from the (hourly) timestamp index.
  - the model is fit on the training set only.
  - the test horizon is produced recursively: each prediction is appended to
    the running history and feeds the lag/rolling features of the next step,
    so there is no test leakage.
  - predictions are clipped to the physical share range [0, 100].
  - confidence band is derived from the std of in-sample residuals.

Reuses the shared layout from the other pipelines:
  - plot_results()  (train | true test | forecast + CI)
  - evaluate()      (MAE, RMSE, MAPE)
  - get_data()      (../data/hub_distribution.csv, two hub columns)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from plot_utils import plot_results

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# FEATURE DEFINITION  (shared with xgboost_pipeline)
# ─────────────────────────────────────────────

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


def build_training_frame(series_train: pd.Series) -> pd.DataFrame:
    s = series_train
    idx = series_train.index
    feats = {f"lag{l}": s.shift(l) for l in LAGS}
    for w in ROLL_WINDOWS:
        feats[f"roll{w}_mean"] = s.shift(1).rolling(w).mean()
    df = pd.DataFrame(feats, index=idx)
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["y"] = s
    return df.dropna()


def feature_row(history: pd.Series, ts: pd.Timestamp) -> np.ndarray:
    # history is indexed by hour and contains all values strictly before `ts`.
    vals = {f"lag{l}": history.iloc[-l] for l in LAGS}
    for w in ROLL_WINDOWS:
        vals[f"roll{w}_mean"] = history.iloc[-w:].mean()
    vals.update(calendar_features(ts))
    return np.array([vals[c] for c in FEATURE_COLS], dtype=float)


# ─────────────────────────────────────────────
# SHARED UTILITIES  (identical layout to xgboost_pipeline.py)
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
# FIT + FORECAST  (LightGBM-specific)
# ─────────────────────────────────────────────

def fit_lightgbm(series_train: pd.Series, params: dict):
    """
    Build the supervised training frame from the training series and fit
    an LGBMRegressor. Returns (model, train_frame).
    """
    train_df = build_training_frame(series_train)
    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["y"].to_numpy()

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    print(f"\n[Model] LightGBM fit on {len(train_df)} supervised rows "
          f"({len(FEATURE_COLS)} features)")
    return model, train_df


def forecast_lightgbm(
    model,
    train_df: pd.DataFrame,
    series_train: pd.Series,
    series_test: pd.Series,
    out_index=None,
    clip_low: float = 0.0,
    clip_high: float = 100.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Recursively forecast the test horizon, feeding each prediction back into
    the lag/rolling features of the next step. Predictions are clipped to the
    physical share range. The 95% band uses the std of in-sample residuals.

    `series_train` / `series_test` carry the hourly DatetimeIndex used for the
    calendar features; `out_index` (defaults to series_test.index) sets the
    index of the returned frame so it can match the original series index.
    """
    history = series_train.copy()
    preds = np.empty(len(series_test), dtype=float)
    for i, ts in enumerate(series_test.index):
        x = feature_row(history, ts).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        yhat = float(np.clip(yhat, clip_low, clip_high))
        preds[i] = yhat
        history = pd.concat([history, pd.Series([yhat], index=[ts])])

    in_sample_pred = model.predict(train_df[FEATURE_COLS].to_numpy())
    resid_std = float(np.std(train_df["y"].to_numpy() - in_sample_pred))
    z = 1.96  # ~95% normal band (alpha=0.05)
    band = z * resid_std

    return pd.DataFrame({
        "forecast": preds,
        "lower_ci": np.clip(preds - band, clip_low, clip_high),
        "upper_ci": np.clip(preds + band, clip_low, clip_high),
    }, index=series_test.index if out_index is None else out_index)


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def lightgbm_pipeline(
    series: pd.Series,
    max_period: int = 365,
    train_ratio: float = 0.8,
    model_params: dict | None = None,
    postfix: str = "",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end LightGBM recursive pipeline with train/test split.
    Structure mirrors xgboost_pipeline().

    Returns
    -------
    (forecast_df, best_params, metrics)
    """
    print("=" * 55)
    print("  LIGHTGBM PIPELINE  (80/20 split, recursive)")
    print("=" * 55)

    default_params = dict(
        n_estimators=600,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    )
    params = {**default_params, **(model_params or {})}

    # ── Keep the original index for the outputs (matches get_data /
    #    the other pipelines); use a DatetimeIndex copy only internally
    #    so the calendar features can be derived. ──
    series_orig = series.astype(float)
    series_dt = series_orig.copy()
    series_dt.index = pd.to_datetime(series_dt.index)

    # ── Train / test split ──────────────────────────────
    split_idx    = round(train_ratio * len(series_orig))
    series_train = series_orig.iloc[:split_idx]   # original index (output/plots)
    series_test  = series_orig.iloc[split_idx:]
    train_dt     = series_dt.iloc[:split_idx]     # DatetimeIndex (modeling)
    test_dt      = series_dt.iloc[split_idx:]
    print(f"[Split] Train: {len(series_train)} obs | Test: {len(series_test)} obs")

    # ── Step 1 — fit on training supervised frame ───────
    model, train_df = fit_lightgbm(train_dt, params)

    # ── Step 2 — recursive forecast for len(test) steps ─
    forecast_df = forecast_lightgbm(model, train_df, train_dt, test_dt,
                                    out_index=series_test.index)

    # ── Step 3 — plot: train | true test | forecast ─────
    plot_results(series_train, series_test, forecast_df,
                 model_name="lightgbm", postfix=postfix)

    # ── Step 4 — metrics ────────────────────────────────
    metrics = evaluate(series_test, forecast_df)

    # ── Print side-by-side comparison ───────────────────
    print("\n[Forecast vs. True]")
    comparison = forecast_df.copy()
    comparison.insert(0, "true", series_test.values)
    print(comparison.to_string())

    best_params = {"feature_cols": FEATURE_COLS, **params}
    return forecast_df, best_params, metrics


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

def get_example_ts() -> pd.Series:
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


def main_lightgbm(ts, postfix):
    forecast_df, best_params, metrics = lightgbm_pipeline(
        ts,
        max_period=50,
        train_ratio=0.8,
        postfix=postfix,
    )
    return forecast_df, best_params, metrics


if __name__ == "__main__":

    df_hub1, df_hub2 = get_data()

    # Hub 1
    forecast_df_hub1, best_params_hub1, metrics_hub1 = lightgbm_pipeline(
        df_hub1,
        max_period=50,
        train_ratio=0.8,
        postfix="hub1",
    )

    # Hub 2
    forecast_df_hub2, best_params_hub2, metrics_hub2 = lightgbm_pipeline(
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

    out_path = Path(__file__).resolve().parent.parent / "data" / "forecast_lightgbm.csv"
    forecast_df.to_csv(out_path)

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolve the project layout relative to this file (machine-independent) ──
# This file lives at <project>/zur_sim/models/calculate_all.py
MODELS_DIR = Path(__file__).resolve().parent          # .../zur_sim/models
DATA_DIR = MODELS_DIR.parent / "data"                 # .../zur_sim/data
PLOTS_DIR = MODELS_DIR.parent / "plots"               # .../zur_sim/plots
PROJECT_DIR = MODELS_DIR.parent.parent                # repo root

# Make the sibling pipeline modules importable regardless of the working dir.
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

# Shared multi-model comparison plots (see plot_utils.py).
from plot_utils import (
    plot_forecasts_combined,
    plot_forecasts_grid,
    plot_metrics_bar,
    plot_residuals,
    plot_error_by_hour,
)

# Import each pipeline's entrypoint defensively: some models depend on heavy
# optional packages (prophet, neuralforecast, torch) that may not be installed
# in every environment. Missing models are skipped instead of aborting the run.
def _load_model(module_name, fn_name):
    try:
        module = __import__(module_name, fromlist=[fn_name])
        return getattr(module, fn_name)
    except Exception as exc:  # ImportError or downstream init error
        print(f"[calculate_all] '{fn_name}' unavailable, skipping "
              f"({type(exc).__name__}: {exc})")
        return None


MODEL_ENTRYPOINTS = {
    # Baselines
    "mean":           _load_model("mean_pipeline",            "main_mean"),
    "naive":          _load_model("naive_pipeline",           "main_naive"),
    "seasonal_naive": _load_model("seasonal_naive_pipeline",  "main_seasonal_naive"),
    "moving_average": _load_model("moving_average_pipeline",  "main_moving_average"),
    # Statistical
    "sarima":         _load_model("sarima_pipeline",          "main_sarima"),
    "ets":            _load_model("ets_pipeline",             "main_ets"),
    "prophet":        _load_model("prophet_pipeline",         "main_prophet"),
    # Machine / deep learning
    "xgboost":        _load_model("xgboost_pipeline",         "main_xgboost"),
    "lightgbm":       _load_model("lightgbm_pipeline",        "main_lightgbm"),
    "dlinear":        _load_model("dlinear_pipeline",         "main_dlinear"),
    "nhits":          _load_model("nhits_pipeline",           "main_nhits"),
    "tft":            _load_model("tft_pipeline",             "main_tft"),
}

def _comparison_metrics(y_true, y_pred):
    """MAE / RMSE / sMAPE in the column names the comparison plots expect."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(100.0 * np.mean(np.where(denom == 0, 0.0, 2.0 * np.abs(err) / denom)))
    return {"MAE": mae, "RMSE": rmse, "sMAPE_%": smape}


def plot_model_comparison(ts, forecasts, label, out_dir=PLOTS_DIR):
    """Compare every model's forecast for one series on the shared test window.

    Parameters
    ----------
    ts : pd.Series
        The full observed series (train + test), indexed by timestamp string.
    forecasts : dict {model_name: forecast_df}
        Each forecast_df has a "forecast" column indexed like the test window
        (the last 20% of `ts`), as returned by each pipeline's main entrypoint.
    label : str
        Used in plot titles and output file names (e.g. "<folder>_<col>").
    """
    forecasts = {k: v for k, v in forecasts.items() if v is not None}
    if not forecasts:
        print(f"[calculate_all] no forecasts to compare for {label}, skipping plots")
        return None

    # Datetime index gives a readable time axis and enables the hour-of-day plot.
    full = ts.copy()
    full.index = pd.to_datetime(full.index)

    # Every model forecasts the same held-out window; size it from any forecast.
    n_test = len(next(iter(forecasts.values())))
    train = pd.DataFrame({"y": full.iloc[:-n_test].to_numpy()}, index=full.index[:-n_test])
    test = pd.DataFrame({"y": full.iloc[-n_test:].to_numpy()}, index=full.index[-n_test:])

    predictions = {name: fc["forecast"].to_numpy() for name, fc in forecasts.items()}
    metrics_df = (
        pd.DataFrame(
            [{"model": name, **_comparison_metrics(test["y"].to_numpy(), yhat)}
             for name, yhat in predictions.items()]
        )
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    # ── Skill vs the flat-mean baseline ─────────────────────────────────────
    # Skill_% = 100 * (1 - model_RMSE / baseline_RMSE).  > 0 means the model
    # genuinely beats "predict the constant mean"; <= 0 means no real skill.
    # Falls back to the actual flat-mean RMSE if the 'mean' model wasn't run.
    if "mean" in metrics_df["model"].values:
        baseline_rmse = float(metrics_df.loc[metrics_df["model"] == "mean", "RMSE"].iloc[0])
    else:
        flat = np.full(len(test), float(train["y"].mean()))
        baseline_rmse = float(np.sqrt(np.mean((test["y"].to_numpy() - flat) ** 2)))
    metrics_df["Skill_%"] = 100.0 * (1.0 - metrics_df["RMSE"] / baseline_rmse)

    print(f"\n[Skill vs flat-mean baseline (RMSE={baseline_rmse:.3f})] for {label}")
    print(metrics_df.to_string(index=False))
    if (metrics_df["Skill_%"] <= 0.5).all():
        print("[calculate_all] WARNING: no model beats the flat mean by >0.5% — "
              "this series has little exploitable signal at this horizon.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_forecasts_combined(train, test, predictions, out_dir / f"compare_{label}_forecasts.png", target=label)
    plot_forecasts_grid(test, predictions, metrics_df, out_dir / f"compare_{label}_grid.png", target=label)
    plot_metrics_bar(metrics_df, out_dir / f"compare_{label}_metrics_bar.png", target=label)
    plot_residuals(test, predictions, out_dir / f"compare_{label}_residuals.png")
    plot_error_by_hour(test, predictions, out_dir / f"compare_{label}_error_by_hour.png", target=label)
    return metrics_df


RESULT_PATH = MODELS_DIR / "result_dict.json"
if RESULT_PATH.exists():
    result_dict = json.load(open(RESULT_PATH))
else:
    result_dict = {

    }

do_break = False
save = False
if __name__ == '__main__':
    for folder in os.listdir(DATA_DIR):
        if folder.endswith("2026-02"):
            df = pd.read_csv(DATA_DIR / folder / 'hub_distribution.csv')
            df.set_index('t', inplace=True)
            for col in df.columns:
                ts = df[col].fillna(0)

                if col == 'Z087':
                # if not col in result_dict:
                    result_dict[col] = {}
                    forecasts = {}
                    for name, main_fn in MODEL_ENTRYPOINTS.items():
                        if main_fn is None:
                            continue  # dependency missing; already reported above
                        forecast_df, best_params, metrics = main_fn(ts, postfix=col)
                        result_dict[col][name] = metrics
                        forecasts[name] = forecast_df

                    # ── Compare all models for this series on one set of plots ──
                    plot_model_comparison(ts, forecasts, label=f"{folder}_{col}")


            if save:
                # do_break = True
                with open(RESULT_PATH, 'w') as outfile:
                    json.dump(result_dict, outfile)

        if do_break:
            break

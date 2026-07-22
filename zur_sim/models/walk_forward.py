"""
Walk-forward (rolling-origin) short-horizon evaluation
======================================================
The single-shot comparison in calculate_all.py forecasts the whole test window
at once, which rewards flat-mean predictors and hides real skill (see
skill_diagnostic.py). This harness evaluates the SAME heavy pipelines the fair
way: at several rolling origins, each forecasting only H steps ahead.

How it reuses the existing pipelines unchanged
----------------------------------------------
Every `*_pipeline(series, ..., train_ratio=, postfix=)` splits at
`round(train_ratio * len(series))`. So to evaluate model M at origin `c` with
horizon `H`, we hand it `series[:c+H]` and `train_ratio = c/(c+H)`: the split
lands exactly on `c`, the model trains on `series[:c]` and forecasts the next
`H` steps, which we score against the held-out `series[c:c+H]`.

Per-origin plotting inside the pipelines is monkey-patched to a no-op and their
chatter is suppressed, so a run produces one clean set of walk-forward plots.

Usage
-----
    python walk_forward.py                      # default models + hubs
    python walk_forward.py --include-neural     # add dlinear/nhits/tft (slow)
    python walk_forward.py sarima ets xgboost   # explicit model subset
"""

import io
import sys
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
DATA_DIR   = MODELS_DIR.parent / "data"
PLOTS_DIR  = MODELS_DIR.parent / "plots"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import plot_utils
from plot_utils import plot_wf_per_origin, plot_wf_box, plot_wf_summary_bars

# ── Configuration ───────────────────────────────────────────────────────────
HORIZON     = 24      # forecast 24h (one day) ahead at each origin
N_ORIGINS   = 6       # number of rolling origins per series
EVAL_START  = 0.65    # first origin sits at 65% of the series; last at len-H

# A high-signal hub (AR beats mean by ~55%) vs the noise hub the user first saw.
DEFAULT_HUBS = [
    ("Z107_Z059_2026-02", "Z107"),   # strong short-horizon signal
    ("Z087_Z059_2026-02", "Z087"),   # near-noise
]

# Fast models run by default; neural nets are opt-in (grid search per origin).
DEFAULT_MODELS = ["mean", "naive", "seasonal_naive", "moving_average",
                  "sarima", "ets", "prophet", "xgboost", "lightgbm"]
NEURAL_MODELS  = ["dlinear", "nhits", "tft"]

PIPELINE_FNS = {
    "mean":           ("mean_pipeline",           "mean_pipeline"),
    "naive":          ("naive_pipeline",          "naive_pipeline"),
    "seasonal_naive": ("seasonal_naive_pipeline", "seasonal_naive_pipeline"),
    "moving_average": ("moving_average_pipeline",  "moving_average_pipeline"),
    "sarima":         ("sarima_pipeline",         "sarima_pipeline"),
    "ets":            ("ets_pipeline",            "ets_pipeline"),
    "prophet":        ("prophet_pipeline",        "prophet_pipeline"),
    "xgboost":        ("xgboost_pipeline",        "xgboost_pipeline"),
    "lightgbm":       ("lightgbm_pipeline",       "lightgbm_pipeline"),
    "dlinear":        ("dlinear_pipeline",        "dlinear_pipeline"),
    "nhits":          ("nhits_pipeline",          "nhits_pipeline"),
    "tft":            ("tft_pipeline",            "tft_pipeline"),
}


def _noop(*args, **kwargs):
    """Do-nothing stand-in used to disable per-origin plotting in pipelines."""
    return None


def _load_pipeline(module_name, fn_name):
    """Import a pipeline fn and silence its per-origin plotting."""
    try:
        module = __import__(module_name, fromlist=[fn_name])
    except Exception as exc:
        print(f"[walk_forward] '{module_name}' unavailable, skipping "
              f"({type(exc).__name__}: {exc})")
        return None
    # Pipelines call plot_results in their own module namespace; stub it out.
    if hasattr(module, "plot_results"):
        module.plot_results = _noop
    return getattr(module, fn_name)


def _metrics(y_true, y_pred):
    """Return MAE, RMSE and symmetric MAPE (%) between truth and forecast."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    denom = np.abs(y_true) + np.abs(y_pred)
    return {
        "MAE":     float(np.mean(np.abs(err))),
        "RMSE":    float(np.sqrt(np.mean(err ** 2))),
        "sMAPE_%": float(100.0 * np.mean(np.where(denom == 0, 0.0,
                                                  2.0 * np.abs(err) / denom))),
    }


def _origins(n):
    """Pick N_ORIGINS rolling cutoffs evenly spaced from EVAL_START*n to n-HORIZON.

    Each returned index c is a train/forecast split point: the model trains on
    series[:c] and forecasts series[c:c+HORIZON]. If the series is too short for
    the window (last cutoff n-HORIZON is at or before the start), only that last
    origin is returned.
    """
    start = round(EVAL_START * n)
    last  = n - HORIZON
    if last <= start:
        return [last]
    return [int(round(x)) for x in np.linspace(start, last, N_ORIGINS)]


def evaluate_series(series: pd.Series, models: dict, label: str) -> pd.DataFrame:
    """Run every model at each rolling origin; return per-origin metrics."""
    n = len(series)
    rows = []
    for c in _origins(n):
        sub = series.iloc[:c + HORIZON]
        tr  = c / len(sub)
        true = series.iloc[c:c + HORIZON].to_numpy(float)
        cutoff_ts = pd.to_datetime(series.index[c - 1])
        print(f"  [{label}] origin @ {cutoff_ts}  (train={c}, h={HORIZON})")
        for name, fn in models.items():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fc_df, _, _ = fn(sub, train_ratio=tr, postfix=f"wf_{label}_{name}")
                yhat = np.asarray(fc_df["forecast"], float)[:HORIZON]
                if len(yhat) < HORIZON:
                    raise ValueError(f"short forecast ({len(yhat)} < {HORIZON})")
                rows.append({"model": name, "cutoff": cutoff_ts, **_metrics(true, yhat)})
            except Exception as exc:
                print(f"      {name}: FAILED ({type(exc).__name__}: {exc})")
    return pd.DataFrame(rows)


def summarize(per_origin: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-origin metrics to per-model mean/std and Skill_%.

    Skill_% = 100 * (1 - model_RMSE_mean / base), where base is the flat 'mean'
    predictor's mean RMSE (falling back to the worst model's RMSE if 'mean' was
    not run). Positive Skill_% means the model beats the flat mean. Rows are
    sorted best (lowest RMSE_mean) first.
    """
    agg = (per_origin.groupby("model")
           .agg(MAE_mean=("MAE", "mean"),   MAE_std=("MAE", "std"),
                RMSE_mean=("RMSE", "mean"),  RMSE_std=("RMSE", "std"),
                sMAPE_mean=("sMAPE_%", "mean"), sMAPE_std=("sMAPE_%", "std"))
           .fillna(0.0).reset_index())
    base = agg.loc[agg["model"] == "mean", "RMSE_mean"]
    base = float(base.iloc[0]) if len(base) else float(agg["RMSE_mean"].max())
    agg["Skill_%"] = 100.0 * (1.0 - agg["RMSE_mean"] / base)
    return agg.sort_values("RMSE_mean").reset_index(drop=True)


def run(model_names, hubs):
    """Walk-forward-evaluate the named models on each hub series and write outputs.

    Loads each pipeline, then for every (folder, column) hub reads its
    hub_distribution.csv, runs evaluate_series + summarize, prints the leaderboard
    with a skill verdict, and saves per-origin/summary CSVs plus walk-forward plots
    to PLOTS_DIR. Missing data files and empty result sets are skipped.
    """
    models = {}
    for name in model_names:
        mod, fn = PIPELINE_FNS[name]
        loaded = _load_pipeline(mod, fn)
        if loaded is not None:
            models[name] = loaded
    print(f"[walk_forward] models: {list(models)}")
    print(f"[walk_forward] horizon={HORIZON}h, origins={N_ORIGINS}, hubs={len(hubs)}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for folder, col in hubs:
        csv = DATA_DIR / folder / "hub_distribution.csv"
        if not csv.exists():
            print(f"[walk_forward] missing {csv}, skipping")
            continue
        s = pd.read_csv(csv).set_index("t")[col].fillna(0)
        label = f"{folder}_{col}"
        print(f"\n=== {label}  ({len(s)} obs) ===")
        per_origin = evaluate_series(s, models, label)
        if per_origin.empty:
            print(f"[walk_forward] no results for {label}")
            continue
        summary = summarize(per_origin)

        print(f"\n[Walk-forward summary] {label}  (h={HORIZON}, {N_ORIGINS} origins)")
        print(summary.to_string(index=False))
        best = summary.iloc[0]
        verdict = ("real skill" if best["Skill_%"] > 1.0 else
                   "NO model beats the flat mean")
        print(f"  -> best: {best['model']}  (Skill_% vs mean = {best['Skill_%']:+.1f}%) "
              f"=> {verdict}")

        per_origin.to_csv(PLOTS_DIR / f"wf_{label}_per_origin.csv", index=False)
        summary.to_csv(PLOTS_DIR / f"wf_{label}_summary.csv", index=False)
        plot_wf_per_origin(per_origin, "RMSE", PLOTS_DIR / f"wf_{label}_rmse_per_origin.png")
        plot_wf_box(per_origin, "RMSE", PLOTS_DIR / f"wf_{label}_rmse_box.png")
        plot_wf_summary_bars(summary, PLOTS_DIR / f"wf_{label}_summary_bars.png", target=label)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    model_names = list(DEFAULT_MODELS)
    if "--include-neural" in flags:
        model_names += NEURAL_MODELS
    if args:                       # explicit model subset overrides defaults
        model_names = args
    run(model_names, DEFAULT_HUBS)

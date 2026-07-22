"""
Apply this project's forecasting models to the UCI Bike Sharing dataset using
the existing walk-forward harness (walk_forward.py) unchanged.

We reuse walk_forward's pipeline loading, rolling-origin evaluation, skill
summary and plots; we only swap the data source to the reshaped bike series
(data/bikeshare_DC_hourly/hub_distribution.csv, see make_bikeshare.py).

By default we evaluate a 4-week window (672h) so the test is apples-to-apples
with the real hub data (~672 rows). Bike rentals have strong daily + weekly
seasonality, so this is a faithful analog of hub demand.

Usage
-----
    python run_bike_walkforward.py                       # cnt, summer 4-week window, fast models
    python run_bike_walkforward.py registered            # the registered/commuter series
    python run_bike_walkforward.py cnt 2012-06-01 1344   # col, start date, n hours
    python run_bike_walkforward.py cnt 2012-06-01 672 --include-neural
"""
import sys
from pathlib import Path

import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import walk_forward as wf
from plot_utils import plot_wf_per_origin, plot_wf_box, plot_wf_summary_bars

DATA_CSV = MODELS_DIR.parent / "data" / "bikeshare_DC_hourly" / "hub_distribution.csv"


def main():
    """Walk-forward-evaluate the models on a bike-share window from the CLI args.

    Parses [col, start, hours] and --include-neural, slices that window from the
    reshaped bike CSV (keeping the string timestamp index the pipelines expect),
    runs walk_forward's evaluate_series/summarize (dropping any non-finite
    results), prints the skill leaderboard, and writes the summary CSVs and
    walk-forward plots to PLOTS_DIR.
    """
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    col   = args[0] if len(args) > 0 else "cnt"
    start = args[1] if len(args) > 1 else "2012-06-01"   # strong-signal summer window
    hours = int(args[2]) if len(args) > 2 else 672        # 4 weeks ~ real hub-data scale

    model_names = list(wf.DEFAULT_MODELS)
    if "--include-neural" in flags:
        model_names += wf.NEURAL_MODELS

    # Load + slice the requested window, keeping the string timestamp index the
    # pipelines expect.
    full = pd.read_csv(DATA_CSV, index_col="t")
    full.index = pd.to_datetime(full.index)
    window = full.loc[start:][[col]].head(hours)
    series = window[col].astype(float)
    series.index = series.index.strftime("%Y-%m-%dT%H:%M:%S")
    label = f"bikeshare_DC_{col}_{start}_{hours}h"

    print(f"[run_bike] series '{col}'  {start} .. +{hours}h  ({len(series)} obs, "
          f"mean={series.mean():.1f}, min={series.min():.0f}, max={series.max():.0f})")

    # Load pipelines (silencing per-origin plotting, exactly like walk_forward).
    models = {}
    for name in model_names:
        mod, fn = wf.PIPELINE_FNS[name]
        loaded = wf._load_pipeline(mod, fn)
        if loaded is not None:
            models[name] = loaded
    print(f"[run_bike] models: {list(models)}")
    print(f"[run_bike] horizon={wf.HORIZON}h, origins={wf.N_ORIGINS}")

    per_origin = wf.evaluate_series(series, models, label)
    if per_origin.empty:
        print("[run_bike] no results")
        return
    # Drop origins where a model diverged to inf/nan (e.g. ETS can fail to fit);
    # otherwise the summary-bar plot's axis limits blow up on non-finite values.
    import numpy as np
    bad = ~np.isfinite(per_origin["RMSE"])
    if bad.any():
        dropped = sorted(per_origin.loc[bad, "model"].unique())
        print(f"[run_bike] dropping non-finite results from: {dropped}")
        per_origin = per_origin[~bad]
    summary = wf.summarize(per_origin)

    print(f"\n[Walk-forward summary] {label}  (h={wf.HORIZON}, {wf.N_ORIGINS} origins)")
    print(summary.to_string(index=False))
    best = summary.iloc[0]
    verdict = "real skill" if best["Skill_%"] > 1.0 else "NO model beats the flat mean"
    print(f"  -> best: {best['model']}  (Skill_% vs mean = {best['Skill_%']:+.1f}%) => {verdict}")

    out = wf.PLOTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    per_origin.to_csv(out / f"wf_{label}_per_origin.csv", index=False)
    summary.to_csv(out / f"wf_{label}_summary.csv", index=False)
    plot_wf_per_origin(per_origin, "RMSE", out / f"wf_{label}_rmse_per_origin.png")
    plot_wf_box(per_origin, "RMSE", out / f"wf_{label}_rmse_box.png")
    plot_wf_summary_bars(summary, out / f"wf_{label}_summary_bars.png", target=label)
    print(f"\n[run_bike] wrote summary + plots to {out}/wf_{label}_*")


if __name__ == "__main__":
    main()

"""
Dense vs sparse (missing-observation) robustness experiment on the bike dataset
===============================================================================
Same question as sparsity_experiment.py — how does each model's forecasting
skill degrade as more hourly observations go missing — but applied to the UCI
Bike Sharing series instead of the Z-hubs.

We reuse sparsity_experiment's machinery unchanged (sparsify -> interpolate ->
fit -> score vs the TRUE window; Skill_% vs the flat mean at the SAME drop-rate)
and only swap the data source to the reshaped bike series
(data/bikeshare_DC_hourly/hub_distribution.csv, see make_bikeshare.py).

By default we use the SAME 4-week (672h) summer window as run_bike_walkforward.py
so the sparsity result is directly comparable to the dense walk-forward run.

Usage
-----
    python run_bike_sparsity.py                       # cnt, summer 4-week window
    python run_bike_sparsity.py registered            # commuter series
    python run_bike_sparsity.py cnt 2012-06-01 672    # col, start, n hours
"""
import sys
from pathlib import Path

import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import walk_forward as wf
import sparsity_experiment as sx

DATA_CSV = MODELS_DIR.parent / "data" / "bikeshare_DC_hourly" / "hub_distribution.csv"


def main():
    args  = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    col   = args[0] if len(args) > 0 else "cnt"
    start = args[1] if len(args) > 1 else "2012-06-01"   # strong-signal summer window
    hours = int(args[2]) if len(args) > 2 else 672        # 4 weeks ~ walk-forward run

    # Optional overrides: --models=a,b  --drops=0.0,0.2  --origins=4  --tag=cheap
    model_list = list(sx.DEFAULT_MODELS)
    tag = ""
    for f in flags:
        if f.startswith("--models="):
            model_list = f.split("=", 1)[1].split(",")
        elif f.startswith("--drops="):
            sx.DROP_RATES = [float(x) for x in f.split("=", 1)[1].split(",")]
        elif f.startswith("--origins="):
            sx.N_ORIGINS = int(f.split("=", 1)[1])
        elif f.startswith("--tag="):
            tag = f.split("=", 1)[1]

    # Load + slice the same window used by run_bike_walkforward, keeping the
    # string timestamp index the pipelines expect.
    full = pd.read_csv(DATA_CSV, index_col="t")
    full.index = pd.to_datetime(full.index)
    window = full.loc[start:][[col]].head(hours)
    series = window[col].astype(float)
    series.index = series.index.strftime("%Y-%m-%dT%H:%M:%S")
    sfx = f"_{tag}" if tag else ""
    label = f"bikeshare_DC_{col}_{start}_{hours}h{sfx}"

    print(f"[bike_sparsity] series '{col}'  {start} .. +{hours}h  ({len(series)} obs, "
          f"mean={series.mean():.1f}, min={series.min():.0f}, max={series.max():.0f})")

    # Same fast, representative model set the hub sparsity sweep uses.
    models = {}
    for name in model_list:
        mod, fn = wf.PIPELINE_FNS[name]
        loaded = wf._load_pipeline(mod, fn)
        if loaded is not None:
            models[name] = loaded
    print(f"[bike_sparsity] models={list(models)} drop_rates={sx.DROP_RATES} "
          f"origins={sx.N_ORIGINS}, horizon={wf.HORIZON}h")

    per = sx.run_hub(series, models, label)
    if per.empty:
        print("[bike_sparsity] no results")
        return
    agg = sx.summarize(per)

    print(f"\n=== {label} — RMSE by drop-rate ===")
    piv = agg.pivot(index="model", columns="drop_rate", values="RMSE").round(2)
    print(piv.sort_values(piv.columns[0]).to_string())
    print(f"\n=== {label} — Skill_% vs flat-mean (same sparsity) ===")
    pivs = agg.pivot(index="model", columns="drop_rate", values="Skill_%").round(1)
    print(pivs.sort_values(pivs.columns[0], ascending=False).to_string())

    out = wf.PLOTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / f"sparsity_{label}_per_origin.csv", index=False)
    agg.to_csv(out / f"sparsity_{label}_summary.csv", index=False)
    if "mean" in agg["model"].values:
        sx.plot_skill_vs_drop(agg, label, out / f"sparsity_{label}.png")
    print(f"\n[bike_sparsity] wrote summary + plot to {out}/sparsity_{label}_*")


if __name__ == "__main__":
    main()

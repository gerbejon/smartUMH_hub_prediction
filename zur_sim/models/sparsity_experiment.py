"""
Sparse vs dense (missing-observation) robustness experiment
===========================================================
Question: how does each model's forecasting skill degrade as the input data
gets sparser — i.e. as more hourly observations go missing?

Method (builds directly on walk_forward.py so results are comparable):
  For each hub, each drop-rate p in DROP_RATES, and each rolling origin c:
    1. take the true training slice series[:c]
    2. randomly drop a fraction p of its interior points, then interpolate
       back onto the full hourly grid (so pipelines that need a regular index
       still run) -> this is the "sparse" input the model trains on
    3. fit the model and forecast HORIZON steps
    4. score the forecast against the TRUE held-out window series[c:c+H]
  Aggregate RMSE / Skill_% across origins, per (hub, model, drop-rate).

p=0.0 is the dense baseline; higher p = sparser. We plot skill vs drop-rate so
"robust to missing data" = a flat line, "fragile" = a line that collapses.

Usage:
    python sparsity_experiment.py                  # default fast model set
    python sparsity_experiment.py sarima xgboost   # explicit subset
"""

import io
import sys
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import walk_forward as wf
from plot_utils import _model_colors
import matplotlib.pyplot as plt

# ── Configuration ───────────────────────────────────────────────────────────
DROP_RATES = [0.0, 0.2, 0.4, 0.6]    # fraction of training points removed
N_ORIGINS  = 4                        # fewer origins than walk_forward (×4 rates)
SEED       = 12345

HUBS = [
    ("Z107_Z059_2026-02", "Z107"),   # strong short-horizon signal
    ("Z087_Z059_2026-02", "Z087"),   # near-noise
]

# Fast, representative set (neural/prophet excluded by default — slow ×4 rates).
DEFAULT_MODELS = ["mean", "seasonal_naive", "sarima", "ets", "xgboost", "lightgbm"]


def sparsify(train: pd.Series, p: float, rng) -> pd.Series:
    """Drop fraction p of interior points, then linearly interpolate back."""
    if p <= 0 or len(train) < 4:
        return train
    interior = np.arange(1, len(train) - 1)        # keep endpoints for interp
    k = int(round(p * len(interior)))
    if k <= 0:
        return train
    drop = rng.choice(interior, size=k, replace=False)
    s = train.astype(float).copy()
    s.iloc[drop] = np.nan
    return s.interpolate(method="linear", limit_direction="both")


def origins(n):
    """Pick N_ORIGINS rolling cutoffs evenly spaced from EVAL_START*n to n-HORIZON.

    Same scheme as walk_forward._origins but with this module's smaller N_ORIGINS
    (fewer origins because each is multiplied across the drop-rate sweep). Falls
    back to just [n-HORIZON] when the series is too short.
    """
    start, last = round(wf.EVAL_START * n), n - wf.HORIZON
    if last <= start:
        return [last]
    return [int(round(x)) for x in np.linspace(start, last, N_ORIGINS)]


def run_hub(series: pd.Series, models: dict, label: str) -> pd.DataFrame:
    """Score every model at each origin and drop-rate against the true window.

    For each rolling origin c and each drop-rate p, sparsifies the true training
    slice series[:c] (drop + interpolate), refits every model on that sparse
    input, forecasts HORIZON steps, and scores the forecast against the untouched
    held-out window series[c:c+HORIZON]. Returns one row per
    (model, drop_rate, origin) with the metric columns.
    """
    h, n = wf.HORIZON, len(series)
    rng = np.random.default_rng(SEED)
    rows = []
    for c in origins(n):
        true = series.iloc[c:c + h].to_numpy(float)
        train_true = series.iloc[:c]
        for p in DROP_RATES:
            train_in = sparsify(train_true, p, rng)
            sub = pd.concat([train_in, series.iloc[c:c + h]])
            tr = len(train_in) / len(sub)
            for name, fn in models.items():
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        fc, _, _ = fn(sub, train_ratio=tr, postfix=f"sp_{label}")
                    yhat = np.asarray(fc["forecast"], float)[:h]
                    if len(yhat) < h:
                        raise ValueError("short forecast")
                    rows.append({"model": name, "drop_rate": p, "cutoff": c,
                                 **wf._metrics(true, yhat)})
                except Exception as exc:
                    print(f"    {name} p={p} @c={c}: FAILED ({type(exc).__name__}: {exc})")
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    """Average metrics per (model, drop-rate) and add within-sparsity Skill_%.

    Skill_% for each row is 100 * (1 - RMSE / base), where base is the flat
    'mean' predictor's RMSE at the SAME drop-rate (fair within-sparsity
    reference), falling back to the worst RMSE if 'mean' is absent.
    """
    agg = (per.groupby(["model", "drop_rate"])
              .agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
                   sMAPE=("sMAPE_%", "mean"))
              .reset_index())
    # Skill vs the flat mean AT THE SAME drop-rate (fair within-sparsity ref).
    base = (agg[agg.model == "mean"].set_index("drop_rate")["RMSE"]
            if (agg.model == "mean").any() else None)
    def skill(r):
        """Skill_% of one row vs the flat mean at the same drop-rate."""
        b = base[r["drop_rate"]] if base is not None else agg["RMSE"].max()
        return 100.0 * (1.0 - r["RMSE"] / b) if b > 0 else 0.0
    agg["Skill_%"] = agg.apply(skill, axis=1)
    return agg


def plot_skill_vs_drop(agg: pd.DataFrame, label: str, out_path: Path):
    """Plot RMSE and Skill_% versus drop-rate (one line per model) to out_path.

    Two side-by-side panels: absolute RMSE and Skill_% vs the flat mean, both
    against the percentage of training observations dropped. A robust model reads
    as a flat line; a fragile one collapses as sparsity rises.
    """
    colors = _model_colors(sorted(agg.model.unique()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, sub in agg.groupby("model"):
        sub = sub.sort_values("drop_rate")
        x = sub["drop_rate"] * 100
        ax1.plot(x, sub["RMSE"], marker="o", label=name, color=colors[name])
        ax2.plot(x, sub["Skill_%"], marker="o", label=name, color=colors[name])
    ax1.set_title("RMSE vs missing-data rate")
    ax1.set_xlabel("% of training observations dropped")
    ax1.set_ylabel("RMSE (mean over origins)")
    ax2.axhline(0, color="#e15759", lw=1)
    ax2.set_title("Skill_% vs flat-mean (same sparsity)")
    ax2.set_xlabel("% of training observations dropped")
    ax2.set_ylabel("Skill_% (>0 = beats mean)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Hub {label} — model robustness to sparse/missing data "
                 f"(h={wf.HORIZON}, {N_ORIGINS} origins)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def main(model_names, tag=""):
    """Run the sparsity sweep for the named models over every hub in HUBS.

    Loads each pipeline, then for each hub reads its series, runs run_hub +
    summarize, prints the RMSE-by-drop-rate table, writes per-origin/summary CSVs
    (optionally tagged), and saves the skill-vs-drop plot when a 'mean' baseline
    is present.
    """
    models = {}
    for name in model_names:
        mod, fn = wf.PIPELINE_FNS[name]
        loaded = wf._load_pipeline(mod, fn)
        if loaded is not None:
            models[name] = loaded
    print(f"[sparsity] models={list(models)} drop_rates={DROP_RATES} "
          f"origins={N_ORIGINS} tag='{tag}'")
    wf.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    for folder, col in HUBS:
        csv = wf.DATA_DIR / folder / "hub_distribution.csv"
        if not csv.exists():
            print(f"[sparsity] missing {csv}, skipping")
            continue
        s = pd.read_csv(csv).set_index("t")[col].fillna(0)
        label = f"{folder}_{col}"
        print(f"\n=== {label} ({len(s)} obs) ===")
        per = run_hub(s, models, label)
        if per.empty:
            print("  no results")
            continue
        agg = summarize(per)
        print(agg.pivot(index="model", columns="drop_rate", values="RMSE")
                 .round(2).to_string())
        per.to_csv(wf.PLOTS_DIR / f"sparsity_{label}{sfx}_per_origin.csv", index=False)
        agg.to_csv(wf.PLOTS_DIR / f"sparsity_{label}{sfx}_summary.csv", index=False)
        # Skill_% is computed vs the mean baseline; only plot when it's present.
        if "mean" in agg["model"].values:
            plot_skill_vs_drop(agg, label, wf.PLOTS_DIR / f"sparsity_{label}{sfx}.png")


if __name__ == "__main__":
    args, tag = [], ""
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
        elif a.startswith("--origins="):
            N_ORIGINS = int(a.split("=", 1)[1])
        elif a.startswith("--drops="):
            DROP_RATES = [float(x) for x in a.split("=", 1)[1].split(",")]
        elif not a.startswith("--"):
            args.append(a)
    main(args if args else DEFAULT_MODELS, tag=tag)

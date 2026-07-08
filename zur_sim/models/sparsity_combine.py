"""
Merge the per-tag sparsity sweeps (cheap / xgb / sarima) into one combined
robustness plot + summary table per hub. Run after the background jobs finish.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import walk_forward as wf
from sparsity_experiment import HUBS, plot_skill_vs_drop, summarize

PLOTS = wf.PLOTS_DIR

for folder, col in HUBS:
    label = f"{folder}_{col}"
    parts = []
    for tag in ("cheap", "xgb", "sarima"):
        f = PLOTS / f"sparsity_{label}_{tag}_per_origin.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    if not parts:
        print(f"[combine] no parts for {label}")
        continue
    per = pd.concat(parts, ignore_index=True)
    # 'mean' appears in every tag — keep one copy.
    per = per.drop_duplicates(subset=["model", "drop_rate", "cutoff"])
    agg = summarize(per)
    out_csv = PLOTS / f"sparsity_{label}_combined_summary.csv"
    agg.to_csv(out_csv, index=False)
    print(f"\n=== {label} (combined) — RMSE by drop-rate ===")
    piv = agg.pivot(index="model", columns="drop_rate", values="RMSE").round(2)
    print(piv.sort_values(piv.columns[0]).to_string())
    plot_skill_vs_drop(agg, label, PLOTS / f"sparsity_{label}_combined.png")

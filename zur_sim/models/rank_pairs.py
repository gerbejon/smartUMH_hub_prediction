"""
Rank all two-hub combinations by walk-forward forecast skill
============================================================
Every dataset in ``data/`` is a two-hub combination: some hub ``Zxxx`` paired
with the fixed hub ``Z059``, stored as two complementary share columns that sum
to ~100. Forecasting the ``Zxxx`` share therefore determines the ``Z059`` share,
so one column is a faithful proxy for the whole pair.

This harness answers "which two-hub combination is best?" the fair way, reusing
the walk-forward machinery in ``walk_forward.py`` unchanged: for every pair it
runs rolling-origin, short-horizon forecasts, computes each model's ``Skill_%``
vs the flat-mean baseline (the reference any real model must beat, see
``eval-methodology``), and records the pair's best model and its skill.

The output is a leaderboard: pairs whose share signal is genuinely forecastable
rise to the top; near-noise pairs (nothing beats the mean) fall to the bottom.

Usage
-----
    python rank_pairs.py                       # fast model set, all pairs
    python rank_pairs.py --models mean,xgboost,lightgbm,seasonal_naive
    python rank_pairs.py --limit 10            # first 10 pairs (smoke test)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from walk_forward import (
    DATA_DIR, PLOTS_DIR, PIPELINE_FNS,
    _load_pipeline, evaluate_series, summarize,
)

# Fast but representative: the flat-mean reference plus the models that carried
# real short-horizon skill on high-signal hubs (see eval-methodology memo).
# Heavy/unstable models (sarima, prophet, ets) are excluded from the 118-pair
# sweep for tractability; re-run walk_forward.py on the winners for the full set.
FAST_MODELS = ["mean", "seasonal_naive", "moving_average", "xgboost", "lightgbm"]


def discover_pairs():
    """Every ``Zxxx_Z059_2026-02`` folder -> (folder_name, forecast_column)."""
    pairs = []
    for d in sorted(DATA_DIR.glob("*_Z059_2026-02")):
        if (d / "hub_distribution.csv").exists():
            hub_a = d.name.split("_Z059")[0]
            pairs.append((d.name, hub_a))
    return pairs


def rank(model_names, limit=None):
    models = {}
    for name in model_names:
        mod, fn = PIPELINE_FNS[name]
        loaded = _load_pipeline(mod, fn)
        if loaded is not None:
            models[name] = loaded
    if "mean" not in models:
        raise SystemExit("the 'mean' baseline is required to compute Skill_%")

    pairs = discover_pairs()
    if limit:
        pairs = pairs[:limit]
    print(f"[rank_pairs] models: {list(models)}")
    print(f"[rank_pairs] ranking {len(pairs)} two-hub pairs "
          f"(each vs Z059), horizon walk-forward\n")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PLOTS_DIR / "pair_ranking.csv"

    rows = []
    t0 = time.time()
    for i, (folder, col) in enumerate(pairs, 1):
        csv = DATA_DIR / folder / "hub_distribution.csv"
        s = pd.read_csv(csv).set_index("t")[col].fillna(0)
        try:
            per_origin = evaluate_series(s, models, folder)
        except Exception as exc:
            print(f"[{i}/{len(pairs)}] {col:5s}  FAILED "
                  f"({type(exc).__name__}: {exc})")
            continue
        if per_origin.empty:
            print(f"[{i}/{len(pairs)}] {col:5s}  no results")
            continue

        summary = summarize(per_origin)          # Skill_% vs flat mean, per model
        real = summary[summary["model"] != "mean"]
        best = real.sort_values("Skill_%", ascending=False).iloc[0]

        row = {
            "pair":        f"{col}+Z059",
            "hub":         col,
            "best_model":  best["model"],
            "best_skill_%": round(float(best["Skill_%"]), 2),
            "best_rmse":   round(float(best["RMSE_mean"]), 3),
            "mean_rmse":   round(float(summary.loc[summary["model"] == "mean",
                                                   "RMSE_mean"].iloc[0]), 3),
        }
        # Per-model skill columns so the leaderboard is self-contained.
        for _, r in real.iterrows():
            row[f"skill_{r['model']}"] = round(float(r["Skill_%"]), 2)
        rows.append(row)

        # Write incrementally so a long run's partial results always survive.
        pd.DataFrame(rows).sort_values("best_skill_%", ascending=False)\
            .to_csv(out_csv, index=False)

        elapsed = time.time() - t0
        eta = elapsed / i * (len(pairs) - i)
        print(f"[{i:3d}/{len(pairs)}] {col:5s}  best={best['model']:14s} "
              f"skill={row['best_skill_%']:+6.1f}%   "
              f"(elapsed {elapsed:4.0f}s, eta {eta:4.0f}s)")

    if not rows:
        raise SystemExit("no pairs produced results")

    board = pd.DataFrame(rows).sort_values("best_skill_%", ascending=False)\
                              .reset_index(drop=True)
    board.to_csv(out_csv, index=False)

    print(f"\n[rank_pairs] leaderboard -> {out_csv}")
    lead_cols = ["pair", "best_model", "best_skill_%", "best_rmse", "mean_rmse"]
    print("\n===== TOP 15 forecastable two-hub combinations =====")
    print(board[lead_cols].head(15).to_string(index=False))
    print("\n===== BOTTOM 5 (near-noise: nothing beats the mean) =====")
    print(board[lead_cols].tail(5).to_string(index=False))

    n_real = int((board["best_skill_%"] > 1.0).sum())
    print(f"\n[rank_pairs] {n_real}/{len(board)} pairs have real skill "
          f"(best model beats flat mean by >1%).")

    _plot_leaderboard(board, PLOTS_DIR / "pair_ranking_top.png")
    print(f"[rank_pairs] plot -> {PLOTS_DIR / 'pair_ranking_top.png'}")
    return board


def _plot_leaderboard(board, out_path, top_n=20):
    top = board.head(top_n).iloc[::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(top)))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(top))))
    ax.barh(top["pair"], top["best_skill_%"], color=colors)
    ax.axvline(0, color="0.4", lw=0.8)
    for y, (skill, model) in enumerate(zip(top["best_skill_%"], top["best_model"])):
        ax.text(skill + 0.4, y, f"{model} {skill:+.0f}%",
                va="center", fontsize=8)
    ax.set_xlabel("Best-model Skill_% vs flat mean  (walk-forward)")
    ax.set_title(f"Top {len(top)} two-hub combinations by forecast skill")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    argv = sys.argv[1:]
    model_names = list(FAST_MODELS)
    limit = None
    for i, a in enumerate(argv):
        if a == "--models" and i + 1 < len(argv):
            model_names = argv[i + 1].split(",")
        if a == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
    rank(model_names, limit=limit)

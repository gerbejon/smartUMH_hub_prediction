"""
Rolling one-step-ahead skill diagnostic
=======================================
The model-comparison plots forecast the *entire* test window in a single
multi-step shot, which structurally rewards models that flatline to the mean.
This script asks the sharper question instead:

    On a fair, rolling ONE-STEP-AHEAD basis, does ANY cheap predictor beat
    "predict the running mean"?  If not, the series has no exploitable
    short-horizon signal and no heavy model can either.

It runs across every hub/column in the 2026-02 datasets (no neural nets, no
grid search — seconds, not hours) and reports the verdict. Predictors:

  mean        running mean of all history before t          (the baseline)
  persistence y[t-1]                                          (naive)
  seasonal24  y[t-24]                                         (daily season)
  ar          linear AR(p) on the last p lags, fit on train

Skill_% = 100 * (1 - predictor_RMSE / mean_RMSE).  > 0 means real skill.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

AR_LAGS     = 24      # AR(p) uses the previous 24 hours
TRAIN_RATIO = 0.8
SEASON      = 24


def _fit_ar(train: np.ndarray, p: int) -> np.ndarray:
    """Least-squares AR(p) with intercept fit on the training values."""
    rows = [train[i - p:i][::-1] for i in range(p, len(train))]
    if not rows:
        return None
    X = np.column_stack([np.ones(len(rows)), np.array(rows)])
    y = train[p:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def one_step_skill(series: np.ndarray) -> dict:
    """Rolling one-step-ahead RMSE for each predictor over the test window."""
    n = len(series)
    split = round(TRAIN_RATIO * n)
    coef = _fit_ar(series[:split], AR_LAGS)

    preds = {k: [] for k in ("mean", "persistence", "seasonal24", "ar")}
    truth = []
    for t in range(split, n):
        hist = series[:t]                      # everything truly known before t
        truth.append(series[t])
        preds["mean"].append(hist.mean())
        preds["persistence"].append(series[t - 1])
        preds["seasonal24"].append(series[t - SEASON] if t >= SEASON else hist.mean())
        if coef is not None and t >= AR_LAGS:
            lags = series[t - AR_LAGS:t][::-1]
            preds["ar"].append(coef[0] + coef[1:] @ lags)
        else:
            preds["ar"].append(hist.mean())

    truth = np.asarray(truth)
    out = {}
    for k, p in preds.items():
        p = np.asarray(p)
        out[f"{k}_RMSE"] = float(np.sqrt(np.mean((truth - p) ** 2)))
        out[f"{k}_MAE"]  = float(np.mean(np.abs(truth - p)))
    base = out["mean_RMSE"]
    for k in ("persistence", "seasonal24", "ar"):
        out[f"{k}_skill_%"] = 100.0 * (1.0 - out[f"{k}_RMSE"] / base) if base > 0 else 0.0
    return out


def run() -> pd.DataFrame:
    rows = []
    folders = sorted(f for f in DATA_DIR.iterdir()
                     if f.is_dir() and f.name.endswith("2026-02"))
    for folder in folders:
        csv = folder / "hub_distribution.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv).set_index("t")
        for col in df.columns:
            s = df[col].fillna(0).to_numpy(dtype=float)
            if len(s) < AR_LAGS * 3:
                continue
            res = one_step_skill(s)
            rows.append({"hub": folder.name, "col": col, **res})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    skill_cols = ["persistence_skill_%", "seasonal24_skill_%", "ar_skill_%"]
    print("\n" + "=" * 64)
    print(f"  ROLLING ONE-STEP-AHEAD SKILL  —  {len(results)} hub/column series")
    print("=" * 64)
    print("\nSkill_% vs running-mean baseline (median across series, >0 = real skill):")
    for c in skill_cols:
        med = results[c].median()
        beats = (results[c] > 1.0).mean() * 100   # % of series beating mean by >1%
        print(f"  {c:<22} median={med:+6.2f}%   beats-mean(>1%) in {beats:4.1f}% of series")

    best = results[skill_cols].max(axis=1)
    print(f"\nBest-of-3 predictor skill: median={best.median():+.2f}%, "
          f"max={best.max():+.2f}%")
    if best.median() <= 1.0:
        print("\nVERDICT: No cheap predictor beats the running mean on a one-step "
              "basis.\n         The hourly share series is ~unpredictable noise — the "
              "model\n         leaderboard mostly ranks 'closeness to the mean', not skill.")
    else:
        print("\nVERDICT: There IS exploitable one-step signal — models that capture "
              "it\n         (AR / lag features) should be preferred over flat baselines.")

    out_csv = PLOTS_DIR / "skill_diagnostic_one_step.csv"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    # Distribution of skill across all series, one box per predictor.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [results[c].to_numpy() for c in skill_cols]
    ax.boxplot(data, tick_labels=[c.replace("_skill_%", "") for c in skill_cols],
               showmeans=True)
    ax.axhline(0, color="#e15759", lw=1.2, label="flat-mean baseline (0% skill)")
    ax.set_ylabel("one-step skill vs running mean (%)")
    ax.set_title("Rolling one-step-ahead skill across all 2026-02 hub series")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png = PLOTS_DIR / "skill_diagnostic_one_step.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    results = run()
    if results.empty:
        print("No series found under", DATA_DIR)
    else:
        summarize(results)

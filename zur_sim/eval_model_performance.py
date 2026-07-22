"""
Forecast Evaluation Metrics
===========================
A self-contained library of point-forecast accuracy metrics used to score model
predictions against the ground truth. Every metric takes two ``pd.Series``
(truth and prediction), inner-joins them on their index so only overlapping
timestamps are compared, and returns a single float.

Metrics are grouped by family: absolute error (MAE, MedAE), squared error
(MSE, RMSE), percentage error (MAPE, sMAPE), scaled/seasonal error (MASE,
RMSSE), goodness-of-fit (R2), directional accuracy, and quantile/pinball loss.
:func:`compute_all_metrics` computes them all at once into a dict. Run as a
script, the module reads the ``forecast_*.csv`` files under ``./data`` and
renders a normalised metric heatmap per hub.
"""

import numpy as np
import pandas as pd
import os
import re

# alignment
def _align_series(y_true: pd.Series, y_pred: pd.Series):
    """Inner-join truth and prediction on their index; return aligned arrays.

    Only timestamps present in both series are kept, so metrics never compare
    mismatched positions. Returns ``(y_true_values, y_pred_values)`` as numpy
    arrays.
    """
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return y_true.values, y_pred.values


# --- Absolute Error Metrics ---

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error: average of ``|y_true - y_pred|``."""
    y_t, y_p = _align_series(y_true, y_pred)
    return np.mean(np.abs(y_t - y_p))


def medae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Median Absolute Error: median of ``|y_true - y_pred|`` (outlier-robust)."""
    y_t, y_p = _align_series(y_true, y_pred)
    return np.median(np.abs(y_t - y_p))


# --- Squared Error Metrics ---

def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Squared Error: average of the squared residuals."""
    y_t, y_p = _align_series(y_true, y_pred)
    return np.mean((y_t - y_p) ** 2)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Squared Error: square root of :func:`mse` (units of the target)."""
    return np.sqrt(mse(y_true, y_pred))


# --- Percentage Metrics ---

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error (%), skipping steps where truth is zero.

    Zero-truth points are masked out to avoid division by zero, so this is
    unreliable on series with many zeros (see :func:`smape`).
    """
    y_t, y_p = _align_series(y_true, y_pred)
    mask = y_t != 0
    return np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric MAPE (%): error normalised by the mean magnitude of truth and
    prediction. Robust to zero-truth points; only steps where both are zero are
    skipped."""
    y_t, y_p = _align_series(y_true, y_pred)
    denom = (np.abs(y_t) + np.abs(y_p)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_t[mask] - y_p[mask]) / denom[mask]) * 100


# --- Scaled Metrics ---

def mase(y_true: pd.Series, y_pred: pd.Series, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error: MAE scaled by the in-sample naive forecast MAE.

    The scale is the mean absolute ``seasonality``-step difference of the truth
    (a naive/seasonal-naive baseline). ``< 1`` means the model beats that naive
    forecast. Returns NaN when the scale is zero (constant series).
    """
    y_t, y_p = _align_series(y_true, y_pred)

    # naive forecast errors
    naive_errors = np.abs(y_t[seasonality:] - y_t[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return np.mean(np.abs(y_t - y_p)) / scale


def rmsse(y_true: pd.Series, y_pred: pd.Series, seasonality: int = 1) -> float:
    """Root Mean Squared Scaled Error: the squared-error analogue of :func:`mase`.

    RMSE scaled by the mean squared ``seasonality``-step naive-forecast error.
    Returns NaN when that scale is zero.
    """
    y_t, y_p = _align_series(y_true, y_pred)

    naive_errors = (y_t[seasonality:] - y_t[:-seasonality]) ** 2
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return np.sqrt(np.mean((y_t - y_p) ** 2) / scale)


# --- Goodness-of-fit ---

def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Coefficient of determination (R^2): fraction of variance explained.

    ``1`` is perfect, ``0`` matches predicting the truth's mean, negative is
    worse than the mean. Returns NaN for a constant truth (zero total variance).
    """
    y_t, y_p = _align_series(y_true, y_pred)

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


# --- Directional Accuracy ---

def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of steps where the forecast gets the direction of change right.

    Compares the sign of consecutive differences of truth vs prediction;
    returns the share of matching up/down/flat moves (1 = always correct).
    """
    y_t, y_p = _align_series(y_true, y_pred)

    true_diff = np.sign(np.diff(y_t))
    pred_diff = np.sign(np.diff(y_p))

    return np.mean(true_diff == pred_diff)


# --- Quantile / Pinball Loss ---

def pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float = 0.5) -> float:
    """Pinball (quantile) loss for quantile level ``q``.

    Penalises under- and over-prediction asymmetrically according to ``q``; at
    ``q=0.5`` it equals half the MAE. Used to score quantile forecasts.
    """
    y_t, y_p = _align_series(y_true, y_pred)
    diff = y_t - y_p
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


# --- Convenience: compute all ---

def compute_all_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute every metric in this module at once.

    Returns a dict keyed by metric name (MAE, MedAE, MSE, RMSE, MAPE, sMAPE,
    MASE, RMSSE, R2, Directional_Accuracy, Pinball_q0.5) for a single
    truth/prediction pair.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "MedAE": medae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred),
        "RMSSE": rmsse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
        "Pinball_q0.5": pinball_loss(y_true, y_pred, q=0.5),
    }

def plot_results(df : pd.DataFrame, title: str):
    """Render a metrics matrix as an annotated heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Metric-by-model table (rows = metrics, columns = models). It is
        transposed before plotting so models run down the y-axis and each cell
        is labelled with its value to two decimals.
    title : str
        Figure title.
    """
    df = df.transpose()
    fig, ax = plt.subplots()
    im = ax.imshow(df.values)

    # ticks
    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[0]))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    # annotate cells
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iloc[i, j]:.2f}",
                    ha="center", va="center", color="black")
    plt.title(title)
    plt.colorbar(im)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    hub_1 = {}
    hub_2 = {}
    for file in os.listdir("./data/"):
        if file.__contains__('forecast'):
            df = pd.read_csv(f"./data/{file}", index_col=0)
            df = df.loc[df.apply(lambda x: all(x.notna()), axis=1),:]
        else:
            continue
        model_name = re.findall(re.escape('forecast_')+r'(.*?)'+re.escape('.csv'), file, flags=re.DOTALL)[0]
        hub_1[model_name] = compute_all_metrics(y_true=df.hub1_y, y_pred=df.hub1_y_hat)
        hub_2[model_name] = compute_all_metrics(y_true=df.hub2_y, y_pred=df.hub2_y_hat)
    df_hub1 = pd.DataFrame(hub_1)
    df_hub1_normalized = df_hub1.div(df_hub1.max(axis=1), axis=0)
    # plot_results(df_hub1, title=f'performance hub1')
    plot_results(df_hub1_normalized, title=f'performance hub1 normalized')

    df_hub2 = pd.DataFrame(hub_2)
    df_hub2_normalized = df_hub2.div(df_hub2.max(axis=1), axis=0)
    plot_results(df_hub2_normalized, title=f'performance hub2 normalized')
    # plot_results(df_hub2, title=f'performance hub2')

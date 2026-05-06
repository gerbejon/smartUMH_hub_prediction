import numpy as np
import pandas as pd
import os
import re

# alignment
def _align_series(y_true: pd.Series, y_pred: pd.Series):
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return y_true.values, y_pred.values


# --- Absolute Error Metrics ---

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)
    return np.mean(np.abs(y_t - y_p))


def medae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)
    return np.median(np.abs(y_t - y_p))


# --- Squared Error Metrics ---

def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)
    return np.mean((y_t - y_p) ** 2)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(mse(y_true, y_pred))


# --- Percentage Metrics ---

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)
    mask = y_t != 0
    return np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)
    denom = (np.abs(y_t) + np.abs(y_p)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_t[mask] - y_p[mask]) / denom[mask]) * 100


# --- Scaled Metrics ---

def mase(y_true: pd.Series, y_pred: pd.Series, seasonality: int = 1) -> float:
    y_t, y_p = _align_series(y_true, y_pred)

    # naive forecast errors
    naive_errors = np.abs(y_t[seasonality:] - y_t[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return np.mean(np.abs(y_t - y_p)) / scale


def rmsse(y_true: pd.Series, y_pred: pd.Series, seasonality: int = 1) -> float:
    y_t, y_p = _align_series(y_true, y_pred)

    naive_errors = (y_t[seasonality:] - y_t[:-seasonality]) ** 2
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return np.sqrt(np.mean((y_t - y_p) ** 2) / scale)


# --- Goodness-of-fit ---

def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


# --- Directional Accuracy ---

def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_t, y_p = _align_series(y_true, y_pred)

    true_diff = np.sign(np.diff(y_t))
    pred_diff = np.sign(np.diff(y_p))

    return np.mean(true_diff == pred_diff)


# --- Quantile / Pinball Loss ---

def pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float = 0.5) -> float:
    y_t, y_p = _align_series(y_true, y_pred)
    diff = y_t - y_p
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


# --- Convenience: compute all ---

def compute_all_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
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

import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests


def get_significant_hubs(df):
    results = []

    for t in df.index:
        scores_at_t = df.loc[t].dropna().values
        all_other_scores = df.drop(t).values.flatten()
        all_other_scores = all_other_scores[~np.isnan(all_other_scores)]

        if len(scores_at_t) == 0:  # skip if no valid values
            results.append({'timestep': t, 'p_value': np.nan})
            continue

        stat, p = stats.mannwhitneyu(scores_at_t, all_other_scores, alternative='less')
        results.append({'timestep': t, 'p_value': p})

    results_df = pd.DataFrame(results)


    rejected, corrected_pvals, _, _ = multipletests(results_df['p_value'], method='fdr_bh')

    results_df['corrected_pval'] = corrected_pvals
    results_df['significant'] = rejected

    return results_df['timestep'].loc[results_df.significant].values


def plot_performance(df, score_name: str):
    significant_hubs = get_significant_hubs(df)

    palette = sns.color_palette(n_colors=df.shape[1])


    # Melt the dataframe to long format (seaborn prefers this)
    df_long = df.reset_index().melt(id_vars='series', var_name='model', value_name=score_name)

    # Plot
    fig, ax = plt.subplots(figsize=(18, 7.5))

    sns.lineplot(data=df_long, x='series', y=score_name, hue='model', ax=ax, palette=palette)

    ax.set_title(f'Model Performance over Observations: {score_name.upper()}')
    ax.set_xlabel('Observation')
    ax.set_ylabel('MAE')
    ax.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0,df_long[score_name].quantile(0.995))
    ax.grid()
    plt.xticks(rotation=90, ha='right')

    if len(significant_hubs) > 0:
        for label in ax.get_xticklabels():
            if label.get_text() in significant_hubs:
                label.set_color('red')


    for i, model in enumerate(df.columns):
        ax.axhline(df[model].mean(), color=palette[i], linestyle='-', linewidth=1, xmin=0, xmax=0.04)
        ax.axhline(df[model].median(), color=palette[i], linestyle='--', linewidth=1, xmin=0.96, xmax=1)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'models/performance/{score_name.upper()}_performance.png')
    plt.show()

def main_performance_significance():
    with open('./models/performance/result_dict.json', 'r') as f:
        data = json.load(f)

    rows = []

    for series_id, models in data.items():
        for model_name, metrics in models.items():
            for metric_name, value in metrics.items():
                rows.append((series_id, model_name, metric_name, value))

    df = pd.DataFrame(rows, columns=["series", "model", "metric", "value"])

    df = df.pivot_table(
        index="series",
        columns=["model", "metric"],
        values="value"
    )

    # df_norm = (df.transpose() / df.transpose().max(axis=1)).transpose()

    df_mape = df[[col for col in df.columns if 'mape' in col]]
    df_mape.columns = df_mape.columns.droplevel('metric')

    df_rmse = df[[col for col in df.columns if 'rmse' in col]]
    df_rmse.columns = df_rmse.columns.droplevel('metric')

    df_mae = df[[col for col in df.columns if 'mae' in col]]
    df_mae.columns = df_mae.columns.droplevel('metric')


    plot_performance(df_mape, score_name='mape')
    plot_performance(df_rmse, score_name='rmse')
    plot_performance(df_mae, score_name='mae')


    res = get_significant_hubs(df_mape)
    res = [item for item in get_significant_hubs(df_mae) if item in res]
    res = [item for item in get_significant_hubs(df_rmse) if item in res]
    print(f'Significant Hubs: {res}')
    #######################333333
    return res


if __name__ == '__main__':
    res = main_performance_significance()
    # dirs = [os.path.join('./data', dir, 'hub_distribution.csv') for dir in os.listdir('./data') if any([dir.__contains__(i) for i in res if i != 'Z059'])]
    dirs = [os.path.join('./data', dir, 'hub_distribution.csv') for dir in os.listdir('./data') if dir.startswith('Z')]

    # ts = pd.read_csv(dirs[0])
    ts_list = []
    for dir in dirs:
        tmp_df = pd.read_csv(dir, index_col=0)
        for col in tmp_df.columns:
            if col != 'Z059':
                ts_list.append(tmp_df[col])

    df = pd.concat(ts_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df.fillna(0, inplace=True)



    fig, ax = plt.subplots(figsize=(12, 5))

    for col in df.columns:
        if col in res:
            ax.plot(df.index, df[col], color='red', alpha=0.4, linewidth=0.8)
        else:
            ax.plot(df.index, df[col], color='black', alpha=0.2, linewidth=0.8)

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.grid()
    plt.tight_layout()
    plt.show()

    #
    # df_long = df.reset_index().melt(id_vars='t', var_name='model', value_name='percentage')
    #
    # # Plot
    # fig, ax = plt.subplots(figsize=(18, 7.5))
    #
    # sns.lineplot(data=df_long, x='t', y='percentage', hue='model', ax=ax)
    #
    # ax.set_title(f'Time Series characteristics')
    # ax.set_xlabel('timestamps')
    # ax.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    # ax.grid()
    # plt.xticks(rotation=90, ha='right')
    # plt.show()
import json
import os
import pandas as pd
import socket
if socket.gethostname() == 'berttrainer-large':
    from sarima_pipeline import main_sarima
    from ets_pipeline import main_ets
    from prophet_pipeline import main_prophet
    from nhits_pipeline import main_nhits
    from tft_pipeline import main_tft
    project_dir = str('/home/ubuntu/smartUMH')

else:
    from zur_sim.models.sarima_pipeline import main_sarima
    from zur_sim.models.ets_pipeline import main_ets
    from zur_sim.models.prophet_pipeline import main_prophet
    from zur_sim.models.nhits_pipeline import main_nhits
    from zur_sim.models.tft_pipeline import main_tft
    project_dir = str('/home/gerj/Documents/playground/smartUMH')

if os.path.exists('result_dict.json'):
    result_dict = json.load(open('result_dict.json'))
else:
    result_dict = {

    }

do_break = False
save = False
if __name__ == '__main__':
    for folder in os.listdir(os.path.join(project_dir, "zur_sim/data")):
        if folder.endswith("2026-02"):
            df = pd.read_csv(os.path.join(project_dir, "zur_sim/data", folder, 'hub_distribution.csv'))
            df.set_index('t', inplace=True)
            for col in df.columns:
                ts = df[col].fillna(0)

                if col == 'Z087':
                # if not col in result_dict:
                    result_dict[col] = {}
                    # SARIMA
                    forecast_df, best_params, metrics = main_sarima(ts, postfix=col)
                    result_dict[col]['sarima'] =  metrics

                    # ETS
                    forecast_df, best_params, metrics = main_ets(ts, postfix=col)
                    result_dict[col]['ets'] = metrics

                    # PROPHET
                    forecast_df, best_params, metrics = main_prophet(ts, postfix=col)
                    result_dict[col]['prophet'] = metrics

                    # NHITS
                    forecast_df, best_params, metrics = main_nhits(ts, postfix=col)
                    result_dict[col]['nhits'] = metrics

                    # TFT
                    forecast_df, best_params, metrics = main_tft(ts, postfix=col)
                    result_dict[col]['tft'] = metrics


            if save:
                # do_break = True
                with open('result_dict.json', 'w') as outfile:
                    json.dump(result_dict, outfile)

        if do_break:
            break

import pandas as pd

from schema_models import ApiPostData
from models import DailyPrediction

daily_df = pd.read_parquet(DailyPrediction.results_path)
# todo: fill dataframes
time_interval_df = pd.DataFrame()
dropout_df = pd.DataFrame()


def prepare_api_response(data: ApiPostData):
    results = {}

    try:
        row = daily_df[(daily_df['date'] == data.date) & (daily_df['PULocationID'] == data.location_id)].iloc[0]
    except IndexError:
        raise
    results['predicted_daily_demand'] = row['pred']

    rows = time_interval_df[
        (time_interval_df['date'] == data.date) & (time_interval_df['PULocationID'] == data.location_id)]
    results['predicted_interval_demand'] = rows.sort_values('interval').to_dict('records')

    rows = dropout_df[
        (dropout_df['date'] == data.date) & (dropout_df['PULocationID'] == data.location_id)]
    results['predicted_dropout'] = rows.sort_values('interval').to_dict('records')

    return results



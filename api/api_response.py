import pandas as pd

from schema_models import ApiPostData
from models import DailyPrediction, IntervalPrediction, DropoutPrediction
from config import high_demands

daily_df = pd.read_parquet(DailyPrediction.results_path)
time_interval_df = pd.read_parquet(IntervalPrediction.results_path)
dropout_df = pd.read_parquet(DropoutPrediction.results_path)


def prepare_api_response(data: ApiPostData):
    results = {}

    try:
        row = daily_df[(daily_df['date'] == data.date) & (daily_df['PULocationID'] == data.location_id)].iloc[0]
    except IndexError:
        raise
    results['predicted_daily_demand'] = row['pred']

    rows = time_interval_df[
        (time_interval_df['date'] == data.date) & (time_interval_df['PULocationID'] == data.location_id)]
    results['predicted_interval_demand'] = rows.sort_values('time_interval_number').to_dict('records')

    if data.location_id in high_demands:
        rows = dropout_df[
            (dropout_df['date'] == data.date) & (dropout_df['PULocationID'] == data.location_id)]
        results['predicted_dropout'] = rows.sort_values('interval').to_dict('records')
    else:
        results['predicted_dropout'] = None

    return results



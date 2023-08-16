import pandas as pd

from schema_models import ApiPostData
from models import DailyPrediction, IntervalPrediction, DropoutPrediction
from config import high_demands

daily_df = pd.read_parquet(DailyPrediction.results_path)
time_interval_df = pd.read_parquet(IntervalPrediction.results_path)
dropout_df = pd.read_parquet(DropoutPrediction.results_path)


def _get_dropout_dicts(df: pd.DataFrame):
    df = df.sort_values('time_interval_number')
    location = df.iloc[0].DOLocationID
    return location, df[['time_interval_number', 'count', 'pred']].to_dict('records')


def prepare_api_response(data: ApiPostData):
    results = {'date': data.date}

    try:
        row = daily_df[(daily_df['date'] == data.date) & (daily_df['PULocationID'] == data.location_id)].iloc[0]
    except IndexError:
        raise
    results['predicted_daily_demand'] = {"pred": int(row['pred'])}

    rows = time_interval_df[
        (time_interval_df['date'] == data.date) & (time_interval_df['PULocationID'] == data.location_id)]
    sorted_results = rows.sort_values('time_interval_number')
    sorted_results['pred'] = sorted_results['pred'].astype(int)

    results['predicted_interval_demand'] = sorted_results[['time_interval_number', 'pred']].to_dict('records')

    if data.location_id in high_demands:
        rows = dropout_df[
            (dropout_df['date'] == data.date) &
            (dropout_df['PULocationID'] == data.location_id) &
            (dropout_df['DOLocationID'].isin(high_demands))]
        print(rows)

        top_data = rows[rows['real demand'] > 20].sort_values(by=['real demand'], ascending=False)

        final_results = top_data.groupby('DOLocationID').apply(_get_dropout_dicts)

        results_dropout = {}
        for do_location_id, res_location in final_results:
            results_dropout[f'{data.location_id}-{do_location_id}'] = res_location

        results['predicted_dropout'] = results_dropout
    else:
        results['predicted_dropout'] = None

    return results



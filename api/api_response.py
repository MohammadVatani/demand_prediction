import pandas as pd

from schema_models import ApiPostData
from models import DailyPrediction

daily_df = pd.read_parquet(DailyPrediction.results_path)
time_interval_df = pd.DataFrame()


def prepare_api_response(data: ApiPostData):
    results = {}

    try:
        row = daily_df[(daily_df['date'] == data.date) & (daily_df['PULocationID'] == data.location_id)].iloc[0]
    except IndexError:
        raise

    results['predicted_daily_demand'] = row['pred']





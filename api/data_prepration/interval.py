from .base import DataPreparation
from config import NUMBER_INTERVAL_PER_DAY
import pandas as pd
from itertools import product
import numpy as np


class IntervalDataPreparation(DataPreparation):

    @classmethod
    def _prepare_data(cls):
        df = cls.load_data()
        interval_hours = int(24 / NUMBER_INTERVAL_PER_DAY)
        df['interval_start'] = df['tpep_pickup_datetime'].dt.floor(f"{interval_hours}H")
        df['interval_end'] = df['interval_start'] + pd.Timedelta(hours=interval_hours)

        df['time_interval'] = df['interval_start'].dt.strftime(
            '%H:%M:%S') + ' - ' + df['interval_end'].dt.strftime('%H:%M:%S')

        df.drop(columns=['interval_start', 'interval_end'], inplace=True)

        df['time_interval_number'] = pd.cut(df['tpep_pickup_datetime'].dt.hour, bins=NUMBER_INTERVAL_PER_DAY,
                                            labels=range(1, NUMBER_INTERVAL_PER_DAY + 1), right=False)
        return df

    @classmethod
    def get_labeled_df(cls):
        df = cls._prepare_data()

        aggregated_df = df.groupby(
            ['date', 'time_interval_number', 'PULocationID']).size().reset_index(name='count')
        unique_dates = df['date'].unique()
        unique_interval = df['time_interval_number'].unique()
        unique_pu_location_ids = df['PULocationID'].unique()
        all_combinations = list(
            product(unique_dates, unique_interval, unique_pu_location_ids))
        combinations_df = pd.DataFrame(all_combinations,
                                       columns=['date', 'time_interval_number', 'PULocationID'])
        label_df = aggregated_df.merge(combinations_df, how='right', on=[
            'date', 'time_interval_number', 'PULocationID']).fillna(0)
        label_df['count'] = label_df['count'] + 1
        label_df.sort_values(by=['PULocationID', 'date', 'time_interval_number'], inplace=True)

        return label_df

    @classmethod
    def add_features(cls, labeled_df: pd.DataFrame):
        df = labeled_df
        df['date'] = df['date'].astype('datetime64[ns]')
        df['PU_day_of_month'] = df['date'].dt.day.astype(np.uint8)
        df['PU_day_of_week'] = df['date'].dt.weekday.astype(np.uint8)
        df = df.sort_values(['PULocationID', 'date', 'time_interval_number'])
        df['last_day_demand'] = df.groupby(['PULocationID'])['count'].shift(NUMBER_INTERVAL_PER_DAY)
        df['last_week_demand'] = df.groupby(['PULocationID'])['count'].shift(NUMBER_INTERVAL_PER_DAY * 7)
        for i in range(1, 7):
            df[f'lag{i}'] = (df.groupby(['PULocationID'])['count'].shift(NUMBER_INTERVAL_PER_DAY * i)) 
        df['label'] = df['count'] / df['last_week_demand']
        df.dropna(inplace=True)
        return df

from .base import DataPreparation
from itertools import product
import pandas as pd
import numpy as np
import pmdarima as pm


class DailyDataPreparation(DataPreparation):
    @classmethod
    def get_labeled_df(cls):
        df = cls.load_data()
        aggregated_df = df.groupby(['date', 'PULocationID']).size().reset_index(name='count')
        unique_dates = df['date'].unique()
        unique_pu_location_ids = df['PULocationID'].unique()
        all_combinations = list(product(unique_dates, unique_pu_location_ids))
        combinations_df = pd.DataFrame(all_combinations, columns=['date', 'PULocationID'])
        labeled_df = aggregated_df.merge(combinations_df, how='right', on=['date', 'PULocationID']).fillna(0)
        labeled_df['count'] = labeled_df['count'] + 1
        labeled_df.sort_values(by=['PULocationID', 'date'], inplace=True)
        return labeled_df

    @classmethod
    def add_features(cls, labeled_df: pd.DataFrame):
        df = labeled_df
        df = pd.merge(df, cls.get_arima_results(df), on=['date', 'PULocationID'])
        df['date'] = df['date'].astype('datetime64')
        df['PU_day_of_month'] = df['date'].dt.day.astype(np.uint8)
        df['week_of_month'] = df['date'].apply(lambda x: (x.day - 1) // 7 + 1)
        df['PU_day_of_week'] = df['date'].dt.weekday.astype(np.uint8)
        df.sort_values(by=['PULocationID', 'date'], inplace=True)
        df['last_day_demand'] = df.groupby(['PULocationID'])['count'].shift(1)
        df['last_week_demand'] = df.groupby(['PULocationID'])['count'].shift(7)

        for i in range(1, 5):
            df[f'lag{i}-{i + 7}'] = (df.groupby(['PULocationID'])['count'].shift(i)) / (
                df.groupby(['PULocationID'])['count'].shift(i + 7))
        df.dropna(inplace=True)
        df['arima'] = df['arima'] / df['last_week_demand']
        df['label'] = df['count'] / df['last_week_demand']
        df.drop(['count'], axis=1, inplace=True)
        return df

    @classmethod
    def get_arima_results(cls, df) -> pd.DataFrame:
        location_dfs = []
        location_ids = df['PULocationID'].unique()
        for location_id in location_ids:
            location_df = df[df['PULocationID'] == location_id].sort_values(
                by=['date']).reset_index(drop=True)
            auto_model = pm.auto_arima(location_df['count'], seasonal=False, max_p=7)
            fitted_values = auto_model.predict_in_sample()
            location_df['pred'] = fitted_values
            location_dfs.append(location_df)
        return pd.concat(location_dfs)






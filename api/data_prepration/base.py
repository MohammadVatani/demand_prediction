import logging
from abc import ABC, abstractmethod
from config import START_TRAINING_DATE, END_TRAINING_DATE
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreparation(ABC):
    data_dir = 'data/input/'

    @classmethod
    def load_data(cls):
        df = pd.read_parquet(cls.data_dir)
        df['date'] = df['tpep_pickup_datetime'].dt.date.astype(str)
        df = df[(df['date'] > START_TRAINING_DATE) & (df['date'] < END_TRAINING_DATE)]
        df = df.sort_values(by='date').reset_index(drop=True)
        return df

    @classmethod
    @abstractmethod
    def add_features(cls, labeled_df):
        pass

    @classmethod
    @abstractmethod
    def get_labeled_df(cls):
        pass

    @classmethod
    def get_featured_data(cls):
        label_df = cls.get_labeled_df()
        logger.info('feature engineering ...')
        return cls.add_features(label_df)





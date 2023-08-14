import xgboost as xgb

from base import Prediction, AbstractModel
from ..config import high_demands, low_demands, mid_demands
from data_prepration import DailyDataPreparation


class Daily:
    name: str
    features = [
        'PU_day_of_week',
        'last_day_demand',
        'last_week_demand',
        'lag1-8',
        'lag2-9',
        'lag3-10',
        'lag4-11',
        'arima',
    ]
    label = 'label'

    @property
    def save_path(self):
        return f'daily/{self.name}/data.parquet'

    def model_class(self):
        return xgb.XGBRegressor


class HighDemandModel(Daily, AbstractModel):
    name = 'high demand daily'
    related_location_ids = high_demands
    model_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
    

class MidDemandModel(Daily, AbstractModel):
    name = 'mid demand daily'
    related_location_ids = mid_demands
    model_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 700}


class LowDemandModel(Daily, AbstractModel):
    name = 'low demand daily'
    related_location_ids = low_demands
    model_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000}
    

class DailyPrediction(Prediction):
    def read_dataset(self):
        return DailyDataPreparation.get_featured_data()

    def read_predict(self, Model, location_id, **time_kwargs):
        location_df = self._cache[Model.name]
        row_item = location_df[location_df['date'] == time_kwargs['date']].iloc[0]
        return row_item[self.pred_field]

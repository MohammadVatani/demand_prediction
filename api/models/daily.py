from .base import Model, Prediction
import xgboost as xgb
from ..config import high_demands, low_demands, mid_demands


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
    save_path = f'daily/{name}/data.parquet'

    def model_class(self):
        return xgb.XGBRegressor


class HighDemandModel(Model, Daily):
    name = 'high demand daily'
    related_location_ids = high_demands
    model_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
    

class MidDemandModel(Model, Daily):
    name = 'mid demand daily'
    related_location_ids = mid_demands
    model_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 700}


class LowDemandModel(Model, Daily):
    name = 'low demand daily'
    related_location_ids = low_demands
    model_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000}
    

class DailyPrediction(Prediction):
    def read_dataset(self):
        pass

    def read_predict(self, Model, location_id, **time_kwargs):
        pass



    
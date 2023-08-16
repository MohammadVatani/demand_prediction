from .base import Prediction, AbstractModel
from data_prepration import IntervalDataPreparation
import xgboost as xgb
from config import high_demands, low_demands, mid_demands


class IntervalModel(AbstractModel):
    features = [
        'time_interval_number',
        'PU_day_of_week',
        'last_day_demand',
        'last_week_demand',
        'lag1',
        'lag2',
        'lag3',
        'lag4',
        'lag5',
        'lag6'
    ]
    label = 'label'

    @property
    def model_class(self):
        return xgb.XGBRegressor

    @property
    def fitted_values(self):
        return self._model.predict(self.x) * self.x['last_week_demand']

    def predict(self, x_test):
        return self._model.predict(x_test) *  x_test['last_week_demand']
    

class IntervalHighDemandModel(IntervalModel):
    name = 'high demand interval'
    model_params = {"learning_rate": 0.1, "max_depth": 3, "n_estimator": 700}
    related_location_ids = high_demands


class IntervalMidDemandModel(IntervalModel):
    name = 'mid demand interval'
    model_params = {"learning_rate": 0.15, "max_depth": 5, "n_estimator": 100}
    related_location_ids = mid_demands


class IntervalLowDemandModel(IntervalModel):
    name = 'low demand interval'
    model_params = {"learning_rate": 0.01, "max_depth": 5, "n_estimator": 700}
    related_location_ids = low_demands


class IntervalPrediction(Prediction):
    target_columns = ['date', 'time_interval_number', 'PULocationID', 'count', 'pred']
    results_path = 'data/results/interval.parquet'
    models = [IntervalHighDemandModel, IntervalMidDemandModel, IntervalLowDemandModel]

    def read_dataset(self):
        return IntervalDataPreparation.get_featured_data()




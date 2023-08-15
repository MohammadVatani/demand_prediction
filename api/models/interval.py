from .base import Prediction, AbstractModel
from data_prepration import IntervalDataPreparation
import xgboost as xgb
from config import high_demands, low_demands, mid_demands


class IntervalModel(AbstractModel):
    name = 'interval_model'
    model_params = {"n_estimator": 700, "max_depth": 5, "learning_rate": 0.1}
    features = [
        'time_interval_number',
        'PU_day_of_week',
        'last_day_demand',
        'last_week_demand',
        'lag1-8',
        'lag2-9',
        'lag3-10',
        'lag4-11',
    ]
    label = 'count'
    related_location_ids = [*low_demands, *mid_demands, *high_demands]

    @property
    def model_class(self):
        return xgb.XGBRegressor

    @property
    def fitted_values(self):
        return self._model.predict(self.x)

    def predict(self, x_test):
        return self._model.predict(x_test)


class IntervalPrediction(Prediction):
    target_columns = ['date', 'time_interval_number', 'PULocationID', 'count', 'pred']
    results_path = 'data/results/interval.parquet'
    models = [IntervalModel]

    def read_dataset(self):
        return IntervalDataPreparation.get_featured_data()




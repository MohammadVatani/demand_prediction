from .base import Prediction, AbstractModel
from data_prepration import DropoffDataPreparation
import xgboost as xgb
from config import high_demands, low_demands, mid_demands


class DropoffModel(AbstractModel):
    name = 'dropout_model'
    model_params = {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}
    features = [
        'PULocationID',
        'DOLocationID',
        'time_interval_number',
        'last_day_demand',
        'last_week_demand',
        'lag1/8',
        'lag2/9',
        'lag3/10',
        'lag4/11'
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


class DropoffPrediction(Prediction):
    target_columns = ['date', 'time_interval_number', 'PULocationID', 'DOLocationID', 'count', 'pred']
    results_path = 'data/results/dropoff.parquet'
    models = [DropoffModel]

    def read_dataset(self):
        return DropoffDataPreparation.get_featured_data()

from .base import Prediction, AbstractModel
from data_prepration import IntervalDataPreparation
import xgboost as xgb
from config import high_demands, low_demands, mid_demands


class IntervalModel(AbstractModel):
    name = 'interval_model'
    model_params = {}
    features = [
    ]
    label = ''
    related_location_ids = [*low_demands, *mid_demands, *high_demands]

    @property
    def model_class(self):
        return xgb.XGBRegressor

    @property
    def fitted_values(self):
        pass

    def predict(self, x_test):
        pass


class IntervalPrediction(Prediction):
    target_columns = ['date', 'PULocationID', 'count', 'pred']
    results_path = 'data/results/interval.parquet'
    models = [IntervalModel]

    def read_dataset(self):
        return IntervalDataPreparation.get_featured_data()




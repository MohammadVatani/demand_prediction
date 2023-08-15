from abc import ABC, abstractmethod
from typing import Type, List
import pandas as pd


class AbstractModel(ABC):
    name: str
    features: list
    label: str
    related_location_ids: list
    save_path: str
    model_params: dict

    @property
    @abstractmethod
    def model_class(self):
        pass

    def get_model_instance(self):
        return self.model_class(**self.model_params)
    
    def __init__(self, df) -> None:
        self.x = df[self.features]
        self.y = df[self.label]
        self._model = self.get_model_instance()

    def fit(self):
        self._model.fit(self.x, self.y)

    @property
    def fitted_values(self):
        return self._model.predict(self.x)

    def predict(self, x_test):
        return self._model.predict(x_test)
        

class Prediction(ABC):
    results_path: str
    interval: str
    models: List[Type[AbstractModel]]
    target_columns: List
    pred_field = 'pred'

    def __init__(self, max_date) -> None:
        self.max_date = max_date

    @classmethod
    def match(cls, model_name):
        return cls.interval == model_name
    
    @abstractmethod
    def read_dataset(self):
        pass
 
    def train_model(self, model_instance, x_train, y_train):
        model_instance.fit(x_train, y_train)
        return model_instance

    def update_predictions(self):
        dataset = self.read_dataset()
        results = []

        for Model in self.models:
            df = dataset[dataset['PULocationID'].isin(Model.related_location_ids)]
            print('feature engineering finished.')
            train_df, test_df = df[df['date'] < self.max_date], df[df['date'] == self.max_date]
            model = Model(train_df)
            model.fit()
            print('model fitted.')
            test_df[self.pred_field] = model.predict(test_df[Model.features])
            results.append(test_df)

        pd.concat(results).to_parquet(self.results_path)


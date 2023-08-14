from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Type, List
import pandas as pd
import os


class Model(ABC):
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
        return self.model_class(self.model_params)
    
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
    base_path_data = './results'
    interval: str
    models: List[Type(Model)]
    target_columns: List
    
    def __init__(self, max_date, cache=False) -> None:
        self.max_date = max_date
        if cache:
            self._cache = self.get_data()

    @classmethod
    def match(cls, model_name):
        return cls.interval == model_name
    
    @abstractmethod
    def read_dataset(self):
        pass

    def get_data(self):
        results = {}
        for Model in self.models:
            path = os.path.join(self.base_path_data, Model.save_path)
            df = pd.read_parquet(path)
            results[Model.name] = df
        return results
 
    def train_model(self, model_instance, x_train, y_train):
        model_instance.fit(x_train, y_train)
        return model_instance
    
    def update_predictions(self):
        dataset = self.read_dataset()

        for Model in self.models:
            df = dataset[dataset['PULocationID'].isin(Model.related_location_ids)]
            train_df, test_df = df[df['date'] <= self.max_date], df[df['date'] > self.max_date]

            model = Model(train_df)
            model.fit()

            train_df['pred'] = model.fitted_values
            test_df['pred'] = model.predict(test_df[Model.features])
            result_df = pd.concat([train_df, test_df])[self.target_columns] 
            result_df.to_parquet(os.path.join(self.base_path_data, model.save_path))
    
    def predict(self, location_id, **time_kwargs):
        if not getattr(self, '_cache'):
            self._cache = self.get_data()   
        Model = self._get_model_for_location(location_id)

        return self.read_predict(Model, location_id, **time_kwargs)
    
    @abstractmethod
    def read_predict(self, Model, location_id, **time_kwargs):
        pass

    def _get_model_for_location(self, location_id) -> Type(Model):
        for Model in self.models:
            if location_id in Model.related_location_ids:
                return Model
        
        raise 'invalid location id'

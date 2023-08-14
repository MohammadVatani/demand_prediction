from abc import ABC, abstractmethod
from typing import Type, List
import pandas as pd
import os


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
    pred_data = None
    base_path_data = './results'
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

    @classmethod
    def get_data(cls):
        results = {}
        for Model in cls.models:
            path = os.path.join(cls.base_path_data, Model.save_path)
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

            train_df[self.pred_field] = model.fitted_values
            test_df[self.pred_field] = model.predict(test_df[Model.features])
            result_df = pd.concat([train_df, test_df])[self.target_columns] 
            result_df.to_parquet(os.path.join(self.base_path_data, model.save_path))
    
    @classmethod
    def predict(cls, location_id, **time_kwargs):
        if not cls.pred_data:
            cls.pred_data = cls.get_data()
        Model = cls._get_model_for_location(location_id)
        return cls.read_predict(Model, location_id, **time_kwargs)
    
    @classmethod
    @abstractmethod
    def read_predict(cls, Model, location_id, **time_kwargs):
        pass

    @classmethod
    def _get_model_for_location(cls, location_id) -> Type[AbstractModel]:
        for Model in cls.models:
            if location_id in Model.related_location_ids:
                return Model
        
        raise 'invalid location id'

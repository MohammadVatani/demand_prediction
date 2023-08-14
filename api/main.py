from fastapi import FastAPI
from schema_models import PredictModel, TrainModel
from models import DailyPrediction


app = FastAPI()


@app.post("/prediction/locations/")
def predict(predict: PredictModel):
    pass


@app.post("/traning/")
def train_models(data: TrainModel):
    mapping = {TrainModel.intervals.TIME_INTERVAL: '', TrainModel.intervals.DAILY: DailyPrediction}

    model_class = mapping.get(data.intervals)
    if model_class:
        model_class(max_date=data.max_date).update_predictions()

    raise 'model not valid'


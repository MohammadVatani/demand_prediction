from fastapi import FastAPI
from schema_models import ApiPostData, TrainModel
from models import DailyPrediction
from api_response import prepare_api_response


app = FastAPI()


mapping = {TrainModel.intervals.TIME_INTERVAL: '', TrainModel.intervals.DAILY: DailyPrediction}


@app.post("/prediction/locations/")
def predict_daily_demand(data: ApiPostData):
    return prepare_api_response(data)


@app.post("/training/")
def train_models(data: TrainModel):
    model_class = mapping.get(data.intervals)
    if model_class:
        model_class(max_date=data.max_date).update_predictions()

    raise 'model not valid'


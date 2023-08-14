from fastapi import FastAPI
from schema_models import ApiPostData, TrainModel
from models import DailyPrediction


app = FastAPI()


mapping = {TrainModel.intervals.TIME_INTERVAL: '', TrainModel.intervals.DAILY: DailyPrediction}


@app.post("/prediction/locations/{location_id}/daily/")
def predict_daily_demand(location_id: int, data: ApiPostData):
    return DailyPrediction.predict(location_id, date=data.date)


@app.post("/prediction/locations/{location_id}/time-interval/")
def predict_time_interval_demand(location_id: int, data: ApiPostData):
    return DailyPrediction.predict(location_id, date=data.date)


@app.post("/traning/")
def train_models(data: TrainModel):
    model_class = mapping.get(data.intervals)
    if model_class:
        model_class(max_date=data.max_date).update_predictions()

    raise 'model not valid'


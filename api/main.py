from fastapi import FastAPI
from schema_models import PredictModel, TrainModel


app = FastAPI()


@app.post("/prediction/locations/")
def predict(predict: PredictModel):
    pass


@app.post("/traning/")
def train_models(data: TrainModel):
    mapping =


import pickle
from typing import Tuple
import FastAPI
import numpy as np

from emulator.inference.data_models import PredictionRequest, PredictionResponse
from emulator.models.model import Model

def load_saved_model(dataset_name: str, model_name: str):
    """
    Load a saved model from disk.
    Unpickle the model and return it.
    """
    model_path = f"{dataset_name}_{model_name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
    
def init_app(dataset_name: str, model_name: str) -> Tuple[Model, FastAPI]:

    model = load_saved_model(dataset_name, model_name)
    return model, FastAPI()

model, app = init_app()

@app.post("/predict")
def predict(req: PredictionRequest) -> PredictionResponse:
    input_array = np.array([req.intpp, req.surface_temp])
    return PredictionResponse(net_biomass = model.predict(input_array))

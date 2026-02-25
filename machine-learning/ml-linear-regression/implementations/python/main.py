import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("../../out/lin_reg_model.joblib")

class PredictRequest(BaseModel):
    sex: float
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array([req.sex, 
                  req.length, 
                  req.diameter, 
                  req.height, 
                  req.whole_weight, 
                  req.shucked_weight, 
                  req.viscera_weight, 
                  req.shell_weight
                ], dtype=float).reshape(1, -1)
    
    pred = model.predict(x)[0]
    return {"prediction": int(pred)}
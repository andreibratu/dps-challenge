import os
import pickle
from logging import getLogger
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel
from xgboost import XGBRegressor
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

logger = getLogger()

# ENV VARIABLES
HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")


# Load the model from GCS
BUCKET = "dsp-challenge-341720-bucket"
try:
    # Try to load from bucket
    with open(f"/gcs/{BUCKET}/model.pkl", "rb") as fp:
        model: XGBRegressor = pickle.load(fp)
    logger.info("Managed to load the trained model from the bucket!")
except:
    with open("model.pkl", "rb") as fp:
        model: XGBRegressor = pickle.load(fp)
    logger.info("Loading from GCS failed, switching to included model..")

# FastAPI HTTP server uses this nice body declarations to parse JSON bodies
class GCPRequest(BaseModel):
    instances: List[List[float]]
    parameters: Optional[Dict] = None


# Set up HTTP server
server = FastAPI()


@server.get(HEALTH_ROUTE, status_code=status.HTTP_200_OK)
def health_endpoint():
    return "OK"


@server.post(PREDICT_ROUTE, status_code=status.HTTP_200_OK)
def predict_endpoint(req: GCPRequest):
    logger.info(f"Received {req} {req.dict()}")
    # We assume the data has already been standardized feature-wise
    X = np.array(req.instances)
    y_hat = model.predict(X).tolist()
    logger.info(f"Predicted: {y_hat}")
    return JSONResponse(content={"predictions": y_hat})

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ml_model import FootballMLPredictor
import pandas as pd

app = FastAPI()

cors_origin = os.getenv("CORS_ORIGIN", "*")
origins = ["*"] if cors_origin == "*" else cors_origin.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = FootballMLPredictor()

class MatchData(BaseModel):
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_form: float
    away_form: float
    h2h_home_wins: int
    h2h_away_wins: int

class PredictionResponse(BaseModel):
    prob_home: float
    prob_draw: float
    prob_away: float
    prob_over_25: float
    prob_under_25: float

class HistoricalData(BaseModel):
    data: List[dict]

@app.post("/api/v1/train")
def train_model(payload: HistoricalData):
    try:
        df = pd.DataFrame(payload.data)
        metrics = predictor.train(df)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_match(match: MatchData):
    try:
        probs = predictor.predict(match.dict())
        return PredictionResponse(**probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/backtest")
def run_backtest(payload: HistoricalData):
    try:
        df = pd.DataFrame(payload.data)
        results = predictor.backtest(df)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
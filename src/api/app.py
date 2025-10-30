from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app=FastAPI(title='California Housing Price Predictor')


#Load trained model
model=joblib.load('model/linear_regression_model.pkl')

#Define Input schema

class InputData(BaseModel):
     MedInc: float
     HouseAge: float
     AveRooms: float
     AveBedrms: float
     Population: float
     AveOccup: float
     Latitude: float
     Longitude: float
     
@app.get('/')
def root():
    return{'message':'Welcome to the California Housing Price Prediction api '}

@app.post('/predict')
def predict(data:InputData):
    df=pd.DataFrame([data.dict()])
    prediction=model.predict(df)[0]
    return {'Predicted Median House Value':round(prediction,4)}
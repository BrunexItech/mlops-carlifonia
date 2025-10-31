from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator


# ========== 1. Setup Logging ==========
os.makedirs("logs", exist_ok=True)  # Create logs directory
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)




app = FastAPI(title='California Housing Price Predictor')

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Load trained model
model = joblib.load('model/linear_regression_model.pkl')

# Define Input schema
class InputData(BaseModel):
     MedInc: float
     HouseAge: float
     AveRooms: float
     AveBedrms: float
     Population: float
     AveOccup: float
     Latitude: float
     Longitude: float

# ========== 2. Track Request Times ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logging.info(f"{request.method} {request.url.path} - {process_time:.2f}ms")
    return response

@app.get('/')
def root():
    logging.info("Home page accessed")
    return {'message':'Welcome to the California Housing Price Prediction api '}

@app.post('/predict')
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]
        
        # Log successful prediction
        logging.info(f"Prediction successful: {round(prediction, 4)}")
        
        return {'Predicted Median House Value': round(prediction, 4)}
    
    except Exception as e:
        # Log errors
        logging.error(f"Prediction failed: {str(e)}")
        return {'error': 'Prediction failed'}
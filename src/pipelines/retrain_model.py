import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Paths - updated for your structure
TRAIN_DATA = "data/splits/train.csv"
NEW_DATA = "data/features/housing_features.csv"
MODEL_PATH = "model/linear_regression_model.pkl"
PREVIOUS_MODEL_PATH = "model/previous_model.pkl"
METRICS_PATH = "metrics/model_performance.json"

def retrain_model():
    """Retrain model when drift is detected"""
    
    # 1️⃣ Load data
    train_df = pd.read_csv(TRAIN_DATA)
    new_df = pd.read_csv(NEW_DATA)
    
    # Combine old + new data
    full_df = pd.concat([train_df, new_df], ignore_index=True)
    
    # 2️⃣ Separate features and target
    X = full_df.drop("MedHouseVal", axis=1)
    y = full_df["MedHouseVal"]
    
    # 3️⃣ Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4️⃣ Train model (using RandomForest like your pipeline)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5️⃣ Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 6️⃣ Load previous performance
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            prev_metrics = json.load(f)
    else:
        prev_metrics = {"r2": 0}
    
    # 7️⃣ Compare & save if improved
    if r2 > prev_metrics["r2"]:
        print(f"✅ New model improved! R²: {r2:.4f} > {prev_metrics['r2']:.4f}")
        
        # Backup current model if exists
        if os.path.exists(MODEL_PATH):
            os.rename(MODEL_PATH, PREVIOUS_MODEL_PATH)
        
        # Save new model
        joblib.dump(model, MODEL_PATH)
        
        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump({"mae": mae, "mse": mse, "r2": r2}, f, indent=4)
        
        return True, r2
    else:
        print(f"❌ New model did not improve. Keeping old model.")
        return False, r2

if __name__ == "__main__":
    retrain_model()
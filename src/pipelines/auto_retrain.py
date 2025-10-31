import pandas as pd
from scipy import stats
import joblib
import os

def check_drift():
    """Check if data drift has occurred"""
    train_df = pd.read_csv("data/splits/train.csv")
    new_df = pd.read_csv("data/features/housing_features.csv")
    
    train_features = train_df.drop('MedHouseVal', axis=1)
    new_features = new_df.drop('MedHouseVal', axis=1)
    
    # Check drift for each feature
    drift_detected = False
    for col in train_features.columns:
        p_value = stats.ks_2samp(train_features[col], new_features[col])[1]
        if p_value < 0.05:  # Drift detected
            drift_detected = True
            break
    
    return drift_detected

def auto_retrain():
    """Automatically retrain model if drift is detected"""
    print("🔍 Checking for data drift...")
    
    drift_detected = check_drift()
    
    if drift_detected:
        print("🚨 Data drift detected! Starting retraining...")
        
        # Import and run retraining
        from retrain_model import retrain_model
        improved, r2_score = retrain_model()
        
        if improved:
            print("✅ Model retrained and improved successfully!")
            return True
        else:
            print("⚠️ Model retrained but no improvement")
            return False
    else:
        print("✅ No data drift detected. No retraining needed.")
        return False

if __name__ == "__main__":
    auto_retrain()
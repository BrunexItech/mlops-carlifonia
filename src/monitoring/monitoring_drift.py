import pandas as pd
from scipy import stats
import datetime
import os

def monitor_drift():
    """Simple drift monitoring using statistical tests"""
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Load data
    train_df = pd.read_csv("data/splits/train.csv")
    new_df = pd.read_csv("data/features/housing_features.csv")
    
    # Remove target column
    train_features = train_df.drop('MedHouseVal', axis=1)
    new_features = new_df.drop('MedHouseVal', axis=1)
    
    # Check drift for each feature using KS test
    drift_results = {}
    for col in train_features.columns:
        p_value = stats.ks_2samp(train_features[col], new_features[col])[1]
        drift_results[col] = p_value < 0.05  # Drift if p-value < 0.05
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"reports/drift_report_{timestamp}.txt", "w") as f:
        for feature, drifted in drift_results.items():
            status = "DRIFT" if drifted else "OK"
            f.write(f"{feature}: {status} (p-value: {stats.ks_2samp(train_features[feature], new_features[feature])[1]:.4f})\n")
    
    # Print summary
    drifted_count = sum(drift_results.values())
    print(f"✅ Drift check completed: {drifted_count}/{len(drift_results)} features drifted")
    
    return drift_results

if __name__ == "__main__":
    monitor_drift()
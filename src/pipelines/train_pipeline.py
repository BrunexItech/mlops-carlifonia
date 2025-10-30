import os 
import sys
import pandas as pd
import joblib
import mlflow 
import mlflow.sklearn


#Add project root to pyhton path
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

from src.data_collection import collect_data
from src.data_cleaning import clean_data
from src.data_validation import validate_data
from src.feature_engineering import create_features
from src.data_split import split_data
from models.train_model import train_model


def run_complete_pipeline():
    """Run the complete ML pipeline from data collection to model training"""
    
    
    #set ML flow Experiment
    mlflow.set_experiment('California_Housing_Regression')
    
    #start MLflow experiment 
    with mlflow.start_run(run_name='Complete_pipeline_run'):
        print('Starting complete ML pipeline with MLflow Tracking...')
        print('='*50)
    
    
        try:
            #Step 1:Data collection
            print('Step 1:Data collection')
            print('-'*30)
            collect_data()
            mlflow.log_param('data_source', 'California_Housing')
            
            #step 2:Data cleaning
            print('\n Step2 : Data Cleaning')
            print('-'*30)
            clean_data()
            mlflow.log_param('Cleaning_method','IQR_outlier_removal')
            
            #Step 3 :Validate Data
            print('step3:Data validation')
            print("-" * 30)
            validation_passed = validate_data()
            if not validation_passed:
                raise Exception('Data validation failed! Pipeline stopped!')
            mlflow.log_param('Validation_passed', True)
            
            
            #step 4 :Feature Engineering
            print('\n STEP 4:Feature Engineering')
            print("-" * 30)
            create_features()
            mlflow.log_param('feature_scaling','StandardScaler')
            
            # Step 5: Data Splitting
            print("\n STEP 5: Data Splitting")
            print("-" * 30)
            split_data() 
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            
            # Step 6: Model Training
            print("\n STEP 6: Model Training")
            print("-" * 30)
            best_model, best_r2 = train_model() 
            
            
            #Log model performance metrics to MLflow
            mlflow.log_metric('best_r2_score', best_r2)
            mlflow.log_param('best_model_type', type(best_model).__name__)
            
            #log the best model
            mlflow.sklearn.log_model(best_model, 'best_model')
            
            #MOdel Registry
            print('\n Step7 :Model Registry')
            print('-'*30)
            
            
            #Get the current run ID
            run_id=mlflow.active_run().info.run_id
            
            
            #Register the model
            mlflow.register_model(model_uri=f'runs:/{run_id}/best_model',
                                  name='CaliforniaHousingModel')
            
            print('MOdel registered successfully')
            
         
         
            
            
            print("\n" + "=" * 50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f" Best Model R² Score: {best_r2:.4f}")
            print('Experiments tracked with MLflow!')
            print("=" * 50)
            
            
            return best_model, best_r2
        
        except Exception as e:
            print(f"\n PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        
if __name__=='__main__':
    run_complete_pipeline()
        
        
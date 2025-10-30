import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib


def train_model(train_path='data/splits/train.csv',
                test_path='data/splits/test.csv',
                model_path='model/linear_regression_model.pkl'):
    
    #Load the train and test splits
    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)
    
    print(f'loaded train data {train_df.shape}')
    print(f'Loaded test_data: {test_df.shape}')
    
    
    #Separate features and target 
    X_train=train_df.drop('MedHouseVal', axis=1)
    y_train=train_df['MedHouseVal']
    X_test=test_df.drop('MedHouseVal', axis=1)
    y_test=test_df['MedHouseVal']
    
    
    print(f'Training features:{X_train.shape}, Train target : {y_train.shape}')
    print(f'Test features : {X_test.shape} , Test target:{y_test.shape}')
    
    
    #Initialize and train multiple models for comparison
    
    models={
        'LinearRegression':LinearRegression(),
        'RandomForest':RandomForestRegressor(n_estimators=100,random_state=42)
        
    }
    
    
    best_model= None
    best_r2 = -float('inf')
    best_model_name = ''
    
    
    print('\n Training models...')
    for name, model in models.items():
        #Train the model
        model.fit(X_train,y_train)
        
        #make predictions
        y_pred=model.predict(X_test)
        
        #Calculate metrics
        mae=mean_absolute_error(y_test,y_pred)
        mse=mean_squared_error(y_test,y_pred)
        r2=r2_score(y_test, y_pred)
        
        
        print(f'\n {name} Evaluation Metrics:')
        print(f'MAE: {mae:.4f}')
        print(f'MSE:{mse:.4f}')
        print(f'R2 Score: {r2:.4f}')
        
        
        #Track the best model
        
        if r2 > best_r2:
            best_r2 = r2
            best_model=model
            best_model_name = name
            
            
    #Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
        
        
        
    print(f'\n Best Model: {best_model_name} with R2 :{best_r2:.4f}')
    print(f'Model saved successfully at :{model_path}')
        
    return best_model, best_r2
    
    
    
if __name__=='__main__':
    train_model()
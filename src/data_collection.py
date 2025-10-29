import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def collect_data(output_path='data/raw/housing.csv'):
    
    #loading the california dataset
    dataset=fetch_california_housing()
    
    #create a dataframe
    df=pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['MedHouseVal']=dataset.target
    
    #Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    #save raw data
    df.to_csv(output_path, index=False)
    print(f'Data saved to {output_path}')
    
    
    
if __name__ == '__main__':
    collect_data()
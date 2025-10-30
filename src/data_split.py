import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(input_path='data/features/housing_features.csv',
               train_path='data/splits/train.csv',
               test_path='data/splits/test.csv',
               test_size=0.2,
               random_state=42):
    df=pd.read_csv(input_path)
    print(f'Loaded features successfully: {df.shape[0]} rows and {df.shape[1]} columns')
    
    
    #split into train/test
    train_df,test_df = train_test_split(df,test_size=test_size, random_state=random_state)
    
    #Create the directories if missing 
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    #save the splits
    train_df.to_csv(train_path,index=False)
    test_df.to_csv(test_path,index=False)
    
    
    print(f'Train data saved to {train_path} ({train_df.shape[0]} rows)')
    print(f'Test data saved to {test_path} ({test_df.shape[0]} rows)')
    
    
if __name__=='__main__':
    split_data()
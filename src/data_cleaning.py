import pandas as pd
import os

def clean_data(input_path='data/raw/housing.csv', output_path='data/processed/housing_clean.csv'):
    #Load raw data
    df=pd.read_csv(input_path)
    print(f'Loaded raw data:{df.shape[0]} rows, {df.shape[1]} columns')
    
    #Handling missing values
    if df.isnull().sum().any():
        df=df.fillna(df.mean())
        print('Missing values filled with colum means')
    else:
        print('No missing values found')
        
        
    #Removing duplicates
    before=df.shape[0]
    df.drop_duplicates(inplace=True)
    after=df.shape[0]
    print(f'Removed {before - after} duplicate rows')
    
    #Handling outliers using IQR (inter quatile range)
    numeric_cols=df.select_dtypes(include=['float64','int64']).columns
    for col in numeric_cols:
        Q1=df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound =Q1 - 1.5 * IQR
        upper_bound=Q3 + 1.5*IQR
        df[col]=df[col].clip(lower_bound,upper_bound)
    print('Outliers handled using IQR method')
    
    #save cleaned data
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print('Cleaned data and saved to {output_path}')


if __name__=='__main__':
    clean_data()
#Preparing data for the model
import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler

def create_features(input_path='data/processed/housing_clean.csv', output_path='data/features/housing_features.csv'):
    #Load the cleaned dataset
    df=pd.read_csv(input_path)
    print(f'Loaded cleaned data: {df.shape}')
    
    #Separate features and targets
    x=df.drop(columns=['MedHouseVal'])
    y=df['MedHouseVal']
    
    
    #Scale numeric features in the dataset 
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(x)
    
    #Convert back to Dataframe
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    
    #Recombine scaled features and the target
    df_scaled = pd.concat([x_scaled,y], axis=1)
    
    
    #save feature-engineering dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True )
    df_scaled.to_csv(output_path, index=False)
    print(f'Feature-engineering data saved to {output_path}')
    
    
if __name__=='__main__':
    create_features()
    
    
    
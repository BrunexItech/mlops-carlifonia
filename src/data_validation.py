import pandas as pd

def validate_data(input_data='data/processed/housing_clean.csv'):
    df=pd.read_csv(input_data)
    print(f'loaded cleaned data:{df.shape[0]} rows, {df.shape[1]} columns')
    
    #Check for missing values
    missing=df.isnull().sum()
    if missing.any():
        print('Missing values found:')
        print(missing[missing > 0])
        return False
    else:
        print('No missing values found')
        
    #Validdating the schema(expected columns)
    expected_columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    
    if list(df.columns)!=expected_columns:
        print('Column schema mismatch!')
        print('Expected:', expected_columns)
        print('Found:',list(df.columns))
        return False
    else:
        print('Schema columns validated')
        
        #check for invalid inputs
        if (df < 0).any().any():
            print('Warning! Some negative values found in the dataset')
        else:
            print('No invalide(negative) values found')
        
        #validate numerical ranges 
        if df["MedHouseVal"].max() > 10 or df["MedHouseVal"].min() < 0:    
            print("⚠️ Target variable seems out of expected range")
        else:
                    print("✅ Target variable within expected range (0–10)")

            
    print("🎯 Data validation passed successfully")
    return True

if __name__=='__main__':
    validate_data()
#%%
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

input_dataframe = pd.read_csv("./HW1/InsuranceCharges.csv")
#%%
# find out that the BMI < 15 or > 50 is nonosense and the data is not valid, so they are outliers
q3, q1 = np.percentile(input_dataframe['bmi'], [75 ,25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
print( 'q1=',q1,' q3=', q3, ' IQR=', q3-q1, ' upper=', upper_band, 'lower=',lower_band)
input_dataframe.loc[(input_dataframe['bmi'] < lower_band) | (input_dataframe['bmi'] > upper_band), 'bmi'] = None
print(input_dataframe)

input_dataframe.isnull().sum() # there is 8 missing values in the 'charges' column

imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(input_dataframe)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=input_dataframe.columns)
print(imputed_dataframe)

##Save Data
imputed_dataframe.to_csv("./HW1/HW1_CleanedDataset.csv",index=False)

##Normalize data
norm_dataframe = (imputed_dataframe - imputed_dataframe.mean()) / imputed_dataframe.std()
print(norm_dataframe)
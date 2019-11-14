import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('/Users/andyliu/Documents/drugdata_students_final.csv')
df_norm = (df - df.mean()) / (df.max() - df.min())
df_norm.to_csv('drug_data_normalized.csv')

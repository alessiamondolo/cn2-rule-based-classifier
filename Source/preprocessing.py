from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
# ZOO DATASET
# Import the dataset as a pandas DataFrame
dataset, meta = arff.loadarff('../Data/arff/zoo.arff')
df = pd.DataFrame(dataset)

# Decoding the dataset, since the string values are in the form u'string_value'
for column in df:
    if df[column].dtype == object:
        df[column] = df[column].str.decode('utf8')

# Coverting 'legs' attribute from float to string
df['legs'] = df['legs'].astype(int).astype(str)

# Removing the 'animal' attribute
df_clean = df.drop('animal', 1)

# Renaming the class column name from 'type' to 'class'
df_clean.rename(columns={'type':'class'}, inplace=True)

# Storing the clean dataset in a .csv file
df_clean.to_csv('../Data/csv/zoo.csv', index=False)

# Creating the train and test datasets and storing them in .csv files
train, test = train_test_split(df_clean, test_size=0.3)
train.to_csv('../Data/csv/zoo_train.csv', index=False)
test.to_csv('../Data/csv/zoo_test.csv', index=False)


# SOYBEAN DATASET
# Import the dataset as a pandas DataFrame
dataset, meta = arff.loadarff('../Data/arff/soybean.arff')
df = pd.DataFrame(dataset)

# Decoding the dataset, since the string values are in the form u'string_value'
for column in df:
    if df[column].dtype == object:
        df[column] = df[column].str.decode('utf8')

# Converting '?' values with NaN
df = df = df.replace('?', np.nan)

# Removing rows with missing values
df_clean = df.dropna()

# Storing the clean dataset in a .csv file
df_clean.to_csv('../Data/csv/soybean.csv', index=False)

# Creating the train and test datasets and storing them in .csv files
train, test = train_test_split(df_clean, test_size=0.3)
train.to_csv('../Data/csv/soybean_train.csv', index=False)
test.to_csv('../Data/csv/soybean_test.csv', index=False)
'''

# IRIS DATASET
# Import the dataset as a pandas DataFrame
dataset, meta = arff.loadarff('../Data/arff/iris.arff')
df = pd.DataFrame(dataset)

# Decoding the dataset, since the string values are in the form u'string_value'
for column in df:
    if df[column].dtype == object:
        df[column] = df[column].str.decode('utf8')

# Converting numerical attributes to categorical attributes (ranges)
sepallength = pd.cut(df['sepallength'], 5).astype(str)
sepalwidth = pd.cut(df['sepalwidth'], 5).astype(str)
petallength = pd.cut(df['petallength'], 5).astype(str)
df['sepallength'] = sepallength
df['sepalwidth'] = sepalwidth
df['petallength'] = petallength

# Storing the clean dataset in a .csv file
df.to_csv('../Data/csv/iris.csv', index=False)

# Creating the train and test datasets and storing them in .csv files
train, test = train_test_split(df, test_size=0.3)
train.to_csv('../Data/csv/iris_train.csv', index=False)
test.to_csv('../Data/csv/iris_test.csv', index=False)
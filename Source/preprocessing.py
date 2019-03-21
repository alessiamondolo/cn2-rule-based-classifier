from scipy.io import arff
import pandas as pd


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

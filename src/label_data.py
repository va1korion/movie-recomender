import pandas as pd
import sklearn
from sklearn import preprocessing
import os
import pickle

dataname = os.environ.get("DATA")
saveto = os.environ.get("FEATURES")
model = os.environ.get("LABELER")

df = pd.read_csv(dataname)
del df['Year']
del df['URL']
del df['Const']
df['Runtime (mins)'].fillna(0, inplace=True)

le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if column_name in ['Directors', 'Genres', 'Title Type', 'Title']:
        df[column_name] = le.fit_transform(df[column_name])
    if column_name in ['Date Rated', 'Release Date']:
        df[column_name] = pd.to_datetime(df[column_name], format='ISO8601')
        df[column_name] = sklearn.preprocessing.minmax_scale(df[column_name])


with open(model, 'wb') as f:
    pickle.dump(le, f)

df_copy, test = sklearn.model_selection.train_test_split(df, train_size=0.85, shuffle=False)
df_copy.to_csv(saveto+'train.csv')
test.to_csv(saveto+'test.csv')

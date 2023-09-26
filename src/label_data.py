import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import pickle

dataname = os.environ.get("DATA",  default="data/data-storage/ratings.csv")
model = os.environ.get("LABELER", default="data/preprocessing/")
train_name = os.environ.get("TRAIN_DATA", default="data/features/train.csv")
test_name = os.environ.get("TEST_DATA", default="data/features/test.csv")

df = pd.read_csv(dataname)
del df['Year']
del df['URL']
del df['Const']
df['Runtime (mins)'].fillna(0, inplace=True)

le = preprocessing.LabelEncoder()
mm = preprocessing.MinMaxScaler()
for column_name in df.columns:
    if column_name in ['Directors', 'Genres', 'Title Type', 'Title']:
        df[column_name] = le.fit_transform(df[column_name])
    if column_name in ['Date Rated', 'Release Date']:
        df[column_name] = pd.to_datetime(df[column_name], format='ISO8601')
        df[column_name] = mm.fit_transform(np.array(df[column_name]).reshape(-1, 1))


with open(model+"label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)


with open(model+"minmaxer.pkl", 'wb') as f:
    pickle.dump(mm, f)

df_copy, test = train_test_split(df, train_size=0.85, shuffle=False)
df_copy.to_csv(train_name)
test.to_csv(test_name)

import pandas as pd
import sklearn
from sklearn import preprocessing
import os
import pickle

dataname = os.environ.get("DATA",  default="data/data-storage/results.csv")
model = os.environ.get("LABELER", default="data/models/label_encoder.pkl")
train_name = os.environ.get("TRAIN_DATA", default="data/features/train.csv")
test_name = os.environ.get("TEST_DATA", default="data/features/test.csv")

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
df_copy.to_csv(train_name)
test.to_csv(test_name)

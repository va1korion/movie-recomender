import numpy as np
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from rfpimp import permutation_importances
import pandas as pd
from datetime import datetime

df = pd.read_csv('src/preprocessing/ratings.csv')
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

print(df.describe())
df_copy, test = sklearn.model_selection.train_test_split(df, train_size=0.85, shuffle=True)

y_train = df_copy['Your Rating']
del df_copy['Your Rating']
X_train = df_copy

random_forest = RandomForestRegressor(criterion='poisson')
random_forest.fit(X_train, y_train)

sgd = LogisticRegression(solver='newton-cholesky')
sgd.fit(X_train, y_train)

mlp = MLPRegressor(hidden_layer_sizes=[20, 20], activation='logistic')
mlp.fit(X_train, y_train)


y = test['Your Rating']
del test['Your Rating']
X = test
Yforest, Ysgd, Ymlp = random_forest.predict(X), sgd.predict(X), mlp.predict(X)

print(f'Forest RMSE: {mean_squared_error(y_true=y, y_pred=Yforest, squared=False)}\n'
      f'Logistic RMSE: {mean_squared_error(y_true=y, y_pred=Ysgd, squared=False)}\n'
      f'MLP RMSE: {mean_squared_error(y_true=y, y_pred=Ymlp, squared=False)}\n')

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import pandas as pd
import os


train_name = os.environ.get("TRAIN_DATA", default="data/features/train.csv")
saveto = os.environ.get("MODEL_ARTIFACT", default="data/models/default_model.pkl")
model = os.environ.get("MODEL", default="RandomForest")


df_copy = pd.read_csv(train_name)
y_train = df_copy['Your Rating']
del df_copy['Your Rating']
X_train = df_copy


if model == "RandomForest":
    model = RandomForestRegressor(criterion='poisson')
elif model == "MLP":
    model = MLPRegressor(hidden_layer_sizes=[20, 20], activation='logistic')
elif model == "Logistic":
    model = LogisticRegression(solver='newton-cholesky')
else:
    exit(4)


model.fit(X_train, y_train)


with open(saveto, 'wb') as f:
    pickle.dump(model, f)

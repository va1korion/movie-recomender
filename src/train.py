import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LogisticRegression
from sklearn.neural_network import MLPRegressor
import pandas as pd
import os


train_name = os.environ.get("TRAIN_DATA", default="data/features/train.csv")
test_name = os.environ.get("TEST_DATA", default="data/features/test.csv")
saveto = os.environ.get("MODEL_ARTIFACT", default="data/models/best_model.pkl")
metrics = os.environ.get("MODEL_METRICS", default="data/models/best_model.json")
model = os.environ.get("MODEL", default="RandomForest")


df_copy = pd.read_csv(train_name)
test = pd.read_csv(test_name)

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
    exit(1)


model.fit(X_train, y_train)

y = test['Your Rating']
del test['Your Rating']
X = test
Ypred = model.predict(X)

with open(metrics, 'w') as f:
    json.dump({
        "RMSE": mean_squared_error(y_true=y, y_pred=Ypred, squared=False)
    }, f)

with open(saveto, 'wb') as f:
    pickle.dump(model, f)

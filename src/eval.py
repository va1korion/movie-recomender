import glob
import pickle
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import json

train_name = os.environ.get("TEST_DATA", default="data/features/test.csv")
modeldir = os.environ.get("MODELS", default="data/models")
saveto = os.environ.get("OPTIMAL_MODEL", default="data/models/best_model.pkl")

min_rmse = 10
test_data = pd.read_csv(train_name)
y = test_data['Your Rating']
del test_data['Your Rating']
X = test_data


def save_best_model(best):
    with open(saveto, 'wb') as f:
        pickle.dump(best, f)


for model in glob.glob(modeldir+"/*.pkl"):
    if "label" not in model:
        with open(model, 'rb') as f:
            m = pickle.load(f)
            Y_pred = m.predict(X)
            rmse = mean_squared_error(y_true=y, y_pred=Y_pred, squared=False)
            if rmse < min_rmse:
                min_rmse = rmse
                save_best_model(m)
                with open(model[:-4]+".json", 'w') as f:
                    json.dump({
                        "RMSE": rmse
                    }, f)

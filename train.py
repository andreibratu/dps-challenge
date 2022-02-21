BUCKET = "dsp-challenge-341720-bucket"

import pickle

# from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score


with open("best.csv", "r") as fp:
    df_final = pd.read_csv(fp)


y = (
    df_final[["mpg"]]
    .to_numpy()
    .reshape(
        -1,
    )
)
X = df_final.drop(columns=["mpg", "Unnamed: 0"]).to_numpy()

##### HYPERPARAMETER SEARCH ATTEMPT

# Define the search space
# hyperparams = {
#     "max_depth": (3, 20),
#     "n_estimators": (10, 500),
#     "learning_rate": (1e-8, 5 * 1e-1, "log-uniform"),
#     "colsample_bytree": (0.1, 1.0),
#     "subsample": (0.1, 1.0),
# }

# Prepare training procedure
model = XGBRegressor(seed=42, random_state=42)
# opt = BayesSearchCV(
#     model,
#     cv=10,
#     n_iter=500,
#     search_spaces=hyperparams,
#     random_state=42,
#     scoring="neg_mean_absolute_error",
#     verbose=3,
#     n_jobs=10,
# )
# We are not test splitting since the optimiser uses cross validation
# opt.fit(X, y)

# This is the best score we could obtain optimizing
# print(np.absolute(opt.best_score_))

# print(opt.best_params_)

model = XGBRegressor(seed=42, random_state=42, colsample_bytree=1)
model.fit(X, y)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(
    model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)
scores = np.absolute(scores)
print("Mean MAE %.3f STD MAE %.3f" % (scores.mean(), scores.std()))

# Dumping the pickled trained model to bucket
with open(f"/gcs/{BUCKET}/model.pkl", "wb+") as fp:
    pickle.dump(model, fp)

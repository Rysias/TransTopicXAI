import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

DATA_DIR = Path("../ExplainlpTwitter/output")


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


pre_model = joblib.load(Path("./models/topic_predictor_v2.joblib"))
logistic = pre_model.steps.pop(-1)
logistic = logistic[1]

all_embs = np.load(DATA_DIR / "topic_embs.npy")

test_embs = pre_model.transform(all_embs[:10, 1:])
np.sigmoi

test_emb = test_embs[0, :]
test_emb.shape

raw_scores = (test_emb.shape * logistic.coef_.T) + logistic.intercept_

top_idx = np.argsort(np.abs(raw_scores), axis=0)[-5:].reshape(-1,)
top_idx
top_scores = raw_scores[
    top_idx,
]

# Getting column names :))
column_names = [str(i) + "wuhu" for i in np.arange(10)]
pre_model[0].get_feature_names(column_names)


plt.barh(np.arange(5), top_scores.reshape(-1,))
plt.show()
np.argmax(np.abs(raw_scores))

sigmoid(np.dot(test_embs, logistic.coef_.T) + logistic.intercept_)
logistic.coef_


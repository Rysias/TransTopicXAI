from typing import Dict
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

DATA_DIR = Path("../ExplainlpTwitter/output")
NEW_DATA_DIR = Path("data/")


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def calc_scores(embeddings: np.ndarray, pre_model, logistic):
    features = pre_model.transform(embeddings)
    return features * logistic.coef_ + logistic.intercept_


def top_n_idx(scores: np.ndarray, n: int = 10) -> np.ndarray:
    return np.argsort(np.abs(scores), axis=1)[0, -n:]


def get_features(polynom, topic_names: Dict[int, str]) -> Dict[int, str]:
    feature_names = polynom.get_feature_names_out(list(topic_names.values()))
    return {i: s.replace(" ", "+") for i, s in enumerate(feature_names)}


def load_models(model_path: Path):
    pre_model = joblib.load(model_path)
    logistic = pre_model.steps.pop(-1)[1]
    full_model = joblib.load(model_path)
    return (pre_model, logistic, full_model)


def predict(embed: np.ndarray, pipe) -> int:
    raw_pred = pipe.predict(embed.reshape(1, -1))[0]
    return "Positive" if raw_pred else "Negative"


def get_colors(scores: np.ndarray) -> str:
    return np.where(scores > 0, "#0057e7", "#d62d20")


def plot_explanation(names, scores, val):
    fig = plt.barh(names, scores, color=get_colors(scores))
    plt.subplots_adjust(left=0.45)
    plt.title(f"Explaning a {val} prediction")
    return fig


def explain_tweet(embed: np.ndarray, model_path: Path):
    pre_model, logistic, full_model = load_models(model_path)
    pred_val = predict(embed, full_model)
    return None


pre_model = joblib.load(Path("./models/topic_predictor_v2.joblib"))
logistic = pre_model.steps.pop(-1)[1]
full_model = joblib.load(Path("./models/topic_predictor_v2.joblib"))

all_embs = np.load(DATA_DIR / "topic_embs.npy")

test_embs = pre_model.transform(all_embs[:10, 1:])

test_emb = test_embs[0, :]

logistic.predict(test_emb.reshape(1, -1))

raw_scores = test_emb * logistic.coef_
intercept_scores = raw_scores + logistic.intercept_

raw_pred = sigmoid(np.sum(raw_scores) + logistic.intercept_)

top_idx = np.argsort(np.abs(intercept_scores), axis=1)[0, -10:]
top_scores = intercept_scores[0, top_idx]

# Getting column names :))
# Names to topic (manually)
topic_names = {
    0: "PopularMedia",
    1: "LocalTalk",
    2: "Media",
    3: "UserSmallTalk",
    4: "SocialExperiences",
    5: "WishesAndDreams",
    6: "NegativeFeelings",
    7: "ExpressingFeelings",
    8: "UserEncouragment",
    9: "UserGreetings",
}


feature_names = pre_model[0].get_feature_names_out(list(topic_names.values()))
feature_name_dict = {i: s.replace(" ", "+") for i, s in enumerate(feature_names)}

top_names = [feature_name_dict[k] for k in top_idx]

color = np.where(top_scores > 0, "#0057e7", "#d62d20")
fig = plt.barh(top_names, top_scores, color=color)
val = "positive"
plt.subplots_adjust(left=0.45)
plt.title(f"Explaning a {val} prediction")
plt.show()

sigmoid(np.dot(test_embs, logistic.coef_.T) + logistic.intercept_)
logistic.coef_

# DATA
test_tweets = pd.read_csv(NEW_DATA_DIR / "bert_test.csv", index_col=0)
test_idx = test_tweets.index
test_embs = all_embs[test_idx, 1:]

full_model.predict(test_embs[0:1, :])[0]

test_tweets["Sentiment"]

predict(test_embs[2, :], full_model)

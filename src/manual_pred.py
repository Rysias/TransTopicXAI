from typing import Dict, List
import joblib
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path("../ExplainlpTwitter/output")
NEW_DATA_DIR = Path("data/")
MODEL_DIR = Path("models")
# Names to topic (manually)
TOPIC_NAMES = {
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


def create_predictor_list() -> List[Path]:
    return [
        model_path
        for model_path in MODEL_DIR.glob("*topic_predictor*")
        if re.match("topic_predictor_\d+\.joblib", model_path.name)
    ]


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def calc_scores(embeddings: np.ndarray, pre_model, logistic):
    features = pre_model.transform(embeddings)
    return features * logistic.coef_


def top_n_idx(scores: np.ndarray, n: int = 10) -> np.ndarray:
    return np.argsort(np.abs(scores), axis=1)[0, -n:]


def get_features(polynom) -> Dict[int, str]:
    feature_names = polynom.get_feature_names_out(list(TOPIC_NAMES.values()))
    return {i: s.replace(" ", "+") for i, s in enumerate(feature_names)}


def get_top_features(polynom, top_indeces):
    feature_names = get_features(polynom)
    return [feature_names[k] for k in top_indeces]


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


def plot_embedding(emb: np.ndarray, pre_model):
    transformed_embs = pre_model.transform(emb.reshape(1, -1))[0, 1:11]
    fig = plt.barh(list(TOPIC_NAMES.values()), transformed_embs)
    plt.subplots_adjust(left=0.45)
    return fig


def plot_explanation(names, scores, val):
    fig = plt.barh(names, scores, color=get_colors(scores))
    plt.subplots_adjust(left=0.45)
    plt.title(f"Explaning a {val} prediction")
    return fig


def explain_tweet(embed: np.ndarray, model_path: Path, n=57):
    embed = embed.reshape(1, -1)
    pre_model, logistic, full_model = load_models(model_path)
    pred_val = predict(embed, full_model)
    scores = calc_scores(embed, pre_model, logistic)
    top_idx = top_n_idx(scores, n=n)
    feature_names = get_top_features(pre_model[0], top_idx)
    top_scores = scores[0, top_idx]
    return plot_explanation(feature_names, top_scores, pred_val)


# DATA
model_list = create_predictor_list()
model_path = create_predictor_list()[-1]
all_embs = np.load(DATA_DIR / "topic_embs.npy")
test_tweets = pd.read_csv(NEW_DATA_DIR / "bert_test.csv", index_col=0)
test_indeces = test_tweets.index
test_embs = all_embs[test_indeces, 1:]
pre_model, logistic, full_model = load_models(model_path)


for test_idx in range(10):
    test_emb = test_embs[test_idx, :]
    tweet_text = test_tweets.loc[test_tweets.index[test_idx], "cleantext"]
    print(f"{tweet_text = }")
    plt.show()
    fig = explain_tweet(test_emb, model_path, n=10)
    plt.show()

# Explore the coefficients
coef_names = pre_model[0].get_feature_names_out(list(TOPIC_NAMES.values()))
coefs = logistic.coef_.reshape(-1)
sort_idx = np.argsort(coefs)

distances = 2 * np.arange(len(coef_names))

plt.barh(distances / 2, coefs[sort_idx], tick_label=coef_names[sort_idx])
plt.gcf().set_size_inches(5, 25)
plt.subplots_adjust(left=0.45, hspace=0)
plt.rc("ytick", labelsize=3)
plt.show()

idx = np.where(coef_names == "UserGreetings")
coefs[idx]

coef_names[10]

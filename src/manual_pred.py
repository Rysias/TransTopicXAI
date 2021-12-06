from typing import Dict, List
import joblib
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt

# from pysentimiento import create_analyzer
from pathlib import Path


DATA_DIR = Path("../ExplainlpTwitter/output")
NEW_DATA_DIR = Path("data/")
MODEL_DIR = Path("models")
SAVE_DIR = NEW_DATA_DIR / "explains"
# Names to topic (manually)




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


def calc_conf(prob: float) -> float:
    conf = prob if (prob > 0.5) else 1 - prob
    return np.round(conf, 2)


def predict(embed: np.ndarray, pipe) -> int:
    raw_proba = pipe.predict_proba(embed.reshape(1, -1))[0][1]
    conf = calc_conf(raw_proba)
    return "Positive" if raw_proba > 0.5 else "Negative", conf


def get_colors(scores: np.ndarray) -> str:
    return np.where(scores > 0, "#0057e7", "#d62d20")


def plot_embedding(emb: np.ndarray, pre_model):
    transformed_embs = pre_model.transform(emb.reshape(1, -1))[0, 1:11]
    fig = plt.barh(list(TOPIC_NAMES.values()), transformed_embs)
    plt.subplots_adjust(left=0.45)
    return fig


def plot_explanation(names, scores, val, conf):
    fig = plt.barh(names, scores, color=get_colors(scores))
    plt.subplots_adjust(left=0.45)
    plt.yticks(fontsize=10)
    plt.title(f"Explaning a {val} with {conf} confidence")
    return fig


def explain_tweet(embed: np.ndarray, model_path: Path, n=57):
    embed = embed.reshape(1, -1)
    pre_model, logistic, full_model = load_models(model_path)
    pred_val, conf = predict(embed, full_model)
    scores = calc_scores(embed, pre_model, logistic)
    top_idx = top_n_idx(scores, n=n)
    feature_names = get_top_features(pre_model[0], top_idx)
    top_scores = scores[0, top_idx]
    return plot_explanation(feature_names, top_scores, val=pred_val, conf=conf)


def create_tweet_name(idx: int) -> Path:
    return SAVE_DIR / f"tweet_{idx}.txt"


def save_tweet(tweet: str, idx: int):
    with open(create_tweet_name(idx), "w") as f:
        f.write(tweet)


def read_json(file_path: Path):
    with open(file_path, "r") as f:
        return json.load(f)


TOPIC_NAMES = {int(k): v for k, v in read_json("tweet_cats.json").items()}

# DATA
model_list = create_predictor_list()
model_path = create_predictor_list()[-1]
all_embs = np.load(DATA_DIR / "topic_embs.npy")
test_tweets = pd.read_csv(NEW_DATA_DIR / "bert_test.csv", index_col=0)
test_indeces = test_tweets.index
test_embs = all_embs[test_indeces, 1:]
pre_model, logistic, full_model = load_models(model_path)


do_save = True
for test_idx in range(10):
    test_emb = test_embs[test_idx, :]
    tweet_idx = test_tweets.index[test_idx]
    tweet_text = test_tweets.loc[tweet_idx, "cleantext"]
    print(f"{tweet_text = }")
    fig = explain_tweet(test_emb, model_path, n=10)
    if not do_save:
        plt.show()
    else:
        plt.savefig(SAVE_DIR / f"topic_exp_{tweet_idx}.png")
        save_tweet(tweet_text, tweet_idx)
    plt.clf()
# Explore the coefficients
coef_names = pre_model[0].get_feature_names_out(list(TOPIC_NAMES.values()))
coefs = logistic.coef_.reshape(-1)
sort_idx = np.argsort(coefs)

distances = 2 * np.arange(len(coef_names))

plt.clf()
plt.barh(
    distances / 2,
    coefs[sort_idx],
    tick_label=coef_names[sort_idx],
    color=get_colors(coefs[sort_idx]),
)
plt.gcf().set_size_inches(10, 15)
plt.subplots_adjust(left=0.45, hspace=0)
plt.rc("ytick", labelsize=15)
plt.title("Global Coefficients")
#plt.savefig(SAVE_DIR / "global_features.png")
plt.show()

full_model.predict(testy[:100, :])
# Getting predictions from pysentimiento #
sentiment = create_analyzer(task="sentiment", lang="en")

preds = sentiment.predict(test_tweets["cleantext"].tolist())
pred_list = [{pred.output: max(pred.probas.values())} for pred in preds]

for tweet_id, pred_dict in zip(test_tweets.index, pred_list):
    with open(SAVE_DIR / f"bertpred_{tweet_id}.json", "w") as f:
        json.dump(pred_dict, f)


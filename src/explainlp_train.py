from typing import Dict, List
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
import train_topic_models as tta
from bertopic import BERTopic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

sys.path.append("C:\\Users\\jhr\\ExplainlpTwitter\\")
from explainlp.clearformer import Clearformer

DATA_DIR = Path("../../BscThesisData/data")
MODEL_PATH = Path("../models")


def read_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_embedding_df(embedding_dict: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    doc_embeddings = {k: emb.mean(axis=0) for k, emb in embedding_dict.items()}
    return pd.DataFrame.from_dict(doc_embeddings, orient="index")


label_df = pd.read_csv(DATA_DIR / "emne_labels.csv")
embedding_paths = list(DATA_DIR.glob("*_embedding_dict.pkl"))
embedding_dict = read_pickle(embedding_paths[0])
embedding_df = create_embedding_df(embedding_dict)
model_name = tta.get_model_name(embedding_paths[0])
topic_df = pd.read_csv(DATA_DIR / f"{model_name}_doc_topics.csv")

# Fitting clearformer
topic_model = BERTopic.load(str(MODEL_PATH / f"{model_name}_topic_model"))

clearformer = Clearformer(topic_model)
embs = embedding_df.dropna().to_numpy()
full_x = np.hstack(
    (
        topic_df["topic"].values.reshape(-1, 1),
        topic_df["prob"].values.reshape(-1, 1),
        embs,
    )
)
clearformer.fit(full_x)


label_df.set_index("case_id", inplace=True)
fit_df = label_df.join(embedding_df).dropna()
y = fit_df["label"]
X = fit_df.drop("label", axis=1).to_numpy()
X_new = clearformer.transform(X)

# Training model
clf = LogisticRegression(class_weight="balanced")
clf.fit(X_new, y)

grid_values = {"C": np.logspace(-2, 2, 20)}
grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring="accuracy")


pipeline = make_pipeline(MinMaxScaler(), grid_clf_acc)
pipeline.fit(X_new, y)

f1_score(y, pipeline.predict(X_new), average=None)
f1_score(y, pipeline.predict(X_new), average="weighted")

pipeline[-1].best_params_

from pprint import pprint

pprint(topic_model.get_topics())

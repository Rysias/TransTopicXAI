import pandas as pd
import pickle
import numpy as np
from explainlp.explainlp import ClearSearch
from pathlib import Path


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def flatten_embeddings(embedding_dict):
    """Creates a big matrix with all the embeddings from the dict"""
    return np.vstack(embedding_dict.values())


model = ClearSearch(
    "Maltehb/-l-ctra-danish-electra-small-cased",
    topic_model=Path("../TransTopicXAI/models/topic_model"),
)


DATA_DIR = Path("../BscThesisData/data")
doc_topics = pd.read_csv(DATA_DIR / "doc_topics.csv")
embeddings_dict = read_pickle(DATA_DIR / "embedding_dict.pkl")
embeddings = flatten_embeddings(embeddings_dict)


model.calculate_centroids(doc_topics["topic"].values, doc_topics["prob"], embeddings)

model.transform_many(embeddings_dict.values())

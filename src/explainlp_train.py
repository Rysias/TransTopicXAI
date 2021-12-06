from typing import Dict, List
import numpy as np
import pandas as pd
import hdbscan
from pathlib import Path
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = next(Path("models").glob("*_topic_model"))
EMB_PATH = Path("data/umap_embs.npy")


doc_path = next(Path("data").glob("*full_doc_topics_*.csv"))
doc_topics = pd.read_csv(doc_path, index_col=0)
cool_idx = doc_topics["topic"] != -1
non_null_topics = doc_topics[cool_idx]["topic"].values

topic_ids, cnts = np.unique(non_null_topics, return_counts=True)


full_embs = np.load(EMB_PATH)
small_embs = full_embs[cool_idx, :]

centroids = np.zeros((len(topic_ids), small_embs.shape[1]))
for i in topic_ids:
    centroids[i, :] += np.mean(small_embs[non_null_topics == i, :], axis=0)

np.save(Path("data/newer_centroids.npy"), centroids)
topic_embs = cosine_similarity(full_embs, centroids)
np.save(Path("data/topic_embs.npy"), topic_embs)


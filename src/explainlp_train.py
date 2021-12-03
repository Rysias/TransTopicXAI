from typing import Dict, List
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = Path("../ExplainlpTwitter/output/topic_model")
EMB_PATH = Path("../ExplainlpTwitter/output/embeddings.npy")

doc_topics = pd.read_csv(Path("data") / "full_doc_topics.csv", index_col=0)
cool_idx = doc_topics["topic"] != -1
non_null_topics = doc_topics[cool_idx]["topic"].values

topic_ids = np.unique(non_null_topics)


full_embs = np.load(EMB_PATH)
small_embs = full_embs[cool_idx, 1:]


centroids = np.zeros((len(topic_ids), small_embs.shape[1]))
for i in topic_ids:
    centroids[i, :] += np.mean(small_embs[non_null_topics == i, :], axis=0)

np.save(Path("data/new_centroids.npy"), centroids)


topic_embs = np.vstack(
    (cosine_similarity(arr, centroids) for arr in np.split(full_embs[:, 1:], 5))
)
np.save(Path("data/topic_embs.npy"), topic_embs)


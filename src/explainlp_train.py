from typing import Dict, List
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from explainlp.clearformer import Clearformer
from bertopic import BERTopic


def latest_pattern(dr: Path, pat: str) -> Path:
    return list(dr.glob(pat))[-1]


def main(args):
    DATA_DIR = Path(args.data_dir)
    doc_topics = latest_pattern(DATA_DIR, "*full_doc_topics_*.csv")
    embeddings = latest_pattern(DATA_DIR, "*bertweet-base-sentiment-analysis_embs.npy")
    topic_model = latest_pattern(DATA_DIR, "*_topic_model")
    clearformer = Clearformer(topic_model)

    filter_idx = doc_topics["topic"] != -1
    non_null_topics = doc_topics.loc[filter_idx, "topic"].values


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(help="creates topic embeddings using Clearformer")
    parser.add_argument(
        "--data-dir",
        default="../../final_proj_dat",
        help="the directory with all the data",
    )
    args = parser.parse_args()
    main(args)

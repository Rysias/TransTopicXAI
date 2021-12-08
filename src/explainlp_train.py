from datetime import datetime
import joblib
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from explainlp.clearformer import Clearformer
from bertopic import BERTopic


def latest_pattern(dr: Path, pat: str) -> Path:
    return list(dr.glob(pat))[-1]


def load_topic_model(dr: Path) -> BERTopic:
    topic_path = latest_pattern(dr, "*_topic_model")
    return BERTopic.load(topic_path)


def load_embeds(dr: Path) -> np.ndarray:
    embedding_path = latest_pattern(dr, "*bertweet-base-sentiment-analysis_embs.npy")
    return np.load(embedding_path)


def load_topics(dr: Path) -> pd.Series:
    doc_topics = latest_pattern(dr, "*full_doc_topics_*.csv")
    return pd.read_csv(doc_topics, usecols=["topic"])["topic"]


def main(args):
    nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = Path(args.data_dir)
    doc_topics = load_topics(DATA_DIR)
    embeddings = load_embeds(DATA_DIR)
    topic_model = load_topic_model(DATA_DIR)
    clearformer = Clearformer(topic_model)

    filter_idx = doc_topics != -1
    non_null_topics = doc_topics.loc[filter_idx].values
    filtered_embeddings = embeddings[filter_idx, :]
    X = np.hstack((non_null_topics.reshape(-1, 1), filtered_embeddings))
    topic_embs = clearformer.fit_transform(X)
    np.save(DATA_DIR / "topic_embs_{nowtime}.npy", topic_embs)
    joblib.dump(clearformer, DATA_DIR / f"clearformer_{nowtime}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(help="creates topic embeddings using Clearformer")
    parser.add_argument(
        "--data-dir",
        default="../../final_proj_dat",
        help="the directory with all the data",
    )
    args = parser.parse_args()
    main(args)

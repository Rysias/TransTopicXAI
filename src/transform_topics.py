"""
Script for transforming embeddings into topics / probs to be used for creating topic embeddings
"""
import argparse
import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path


def load_embeddings(DATA_DIR: Path) -> np.ndarray:
    path = DATA_DIR / "embeddings.npy"
    return np.load(path)[:, 1:]


def load_docs(DATA_DIR: Path) -> np.ndarray:
    path = DATA_DIR / "clean_tweets.csv"
    return pd.read_csv(path, usecols=["cleantweets"])["cleantweets"]


def load_topic_model(path: Path) -> BERTopic:
    return BERTopic.load(str(path))


def main(args):
    DATA_DIR = Path(args.data_dir)
    topic_model = load_topic_model(Path(args.topic_model))
    embeddings = load_embeddings(DATA_DIR)
    docs = load_docs(DATA_DIR)
    topics, probs = topic_model.transform(docs, embeddings)
    full_doc_topics = pd.DataFrame({"topic": topics, "prob": probs})
    full_doc_topics.to_csv(DATA_DIR / "full_doc_topics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for transforming embeddings into topics")
    parser.add_argument("--data-dir", type=str, help="Directory to find and save data")
    parser.add_argument("--topic-model", type=str, help="Path to topic model")
    args = parser.parse_args()
    main(args)

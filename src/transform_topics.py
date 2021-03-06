"""
Script for transforming embeddings into topics / probs to be used for creating topic embeddings
"""
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from bertopic import BERTopic
from pathlib import Path


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%d-%m-%Y:%H:%M:%S",
)


def load_embeddings(DATA_DIR: Path) -> np.ndarray:
    path = DATA_DIR / "bertweet-base-sentiment-analysis_embs.npy"
    return np.load(path)[:, 1:]


def load_docs(DATA_DIR: Path) -> np.ndarray:
    path = DATA_DIR / "tweeteval_text.csv"
    return pd.read_csv(path, usecols=["text"])["text"]


def load_topic_model(DATA_DIR: Path) -> BERTopic:
    topic_path = str(list(DATA_DIR.glob("*_topic_model"))[-1])
    logging.debug(f"{topic_path = }")
    return BERTopic.load(topic_path)


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = Path(args.data_dir)
    logging.info("loading topic model...")
    topic_model = load_topic_model(DATA_DIR)
    logging.info("loading embeddings")
    embeddings = load_embeddings(DATA_DIR)
    logging.info("loading data")
    docs = load_docs(DATA_DIR)
    logging.info("transforming embeddings / docs")
    topics, probs = topic_model.transform(docs, embeddings)
    logging.info("writing to file!")
    full_doc_topics = pd.DataFrame({"topic": topics, "prob": probs})
    full_doc_topics.to_csv(DATA_DIR / f"full_doc_topics_{timestamp}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for transforming embeddings into topics")
    parser.add_argument("--data-dir", type=str, help="Directory to find and save data")
    args = parser.parse_args()
    main(args)

from datetime import datetime
import joblib
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from explainlp.clearformer import Clearformer
from bertopic import BERTopic


def latest_pattern(dr: Path, pat: str) -> Path:
    return sorted(dr.glob(pat))[-1]


def load_topic_model(dr: Path) -> BERTopic:
    topic_path = latest_pattern(dr, "*_topic_model")
    return BERTopic.load(topic_path)


def load_embeds(dr: Path) -> np.ndarray:
    embedding_path = latest_pattern(dr, "*bertweet-base-sentiment-analysis_embs.npy")
    return np.load(embedding_path)


def load_topics(dr: Path) -> pd.Series:
    doc_topics = latest_pattern(dr, "*full_doc_topics_*.csv")
    print(doc_topics)
    return pd.read_csv(doc_topics, usecols=["topic"])["topic"]


def get_filter_idx(doc_topics: pd.Series, test_idx: pd.Index, get_testset=False):
    not_in_test = ~doc_topics.index.isin(test_idx)
    if get_testset:
        return ~not_in_test
    is_topic = doc_topics != -1
    return not_in_test & is_topic


def main(args):
    nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = Path(args.data_dir)
    test_idx = pd.read_csv(DATA_DIR / "tweeteval_test.csv", index_col=0).index
    doc_topics = load_topics(DATA_DIR)
    embeddings = load_embeds(DATA_DIR)
    topic_model = load_topic_model(DATA_DIR)
    clearformer = Clearformer(topic_model)

    filter_idx = get_filter_idx(doc_topics, test_idx)
    non_null_topics = doc_topics.loc[filter_idx].values
    filtered_embeddings = embeddings[filter_idx, :]
    print(f"{filtered_embeddings.shape = }")
    print(f"{non_null_topics.shape = }")
    X = np.hstack((non_null_topics.reshape(-1, 1), filtered_embeddings))
    topic_embs = clearformer.fit_transform(X)
    # Create test set embeddings
    test_filter = get_filter_idx(doc_topics, test_idx, get_testset=True)
    test_embs = clearformer.transform(embeddings[~test_filter, :])
    # Save it all to files
    np.save(DATA_DIR / "topic_embs_train_{nowtime}.npy", topic_embs)
    np.save(DATA_DIR / "topic_embs_test_{nowtime}.npy", test_embs)
    joblib.dump(clearformer, DATA_DIR / f"clearformer_{nowtime}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="creates topic embeddings using Clearformer"
    )
    parser.add_argument(
        "--data-dir",
        default="../../final_proj_dat",
        help="the directory with all the data",
    )
    args = parser.parse_args()
    main(args)

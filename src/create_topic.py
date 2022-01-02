"""
Trains topic model on pseudo-random subset of the training data
NB: HDBSCAN breaks with more than ~40k documents, so keep it below that
Writes the topic model as a file as well as the topic words from the c-TF-IDF to a pickle
"""
from datetime import datetime
from pathlib import Path
import pickle
from typing import Any, Union
import logging
import numpy as np
import pandas as pd
import argparse
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

try:
    from cuml.manifold.umap import UMAP
except ModuleNotFoundError:
    from umap import UMAP


def load_docs(data_path: Union[str, Path], text_col="text") -> np.ndarray:
    df = pd.read_csv(Path(data_path))
    return df[text_col]


def load_embeds(embed_path: Union[str, Path]):
    return np.load(embed_path)


def get_random_idx(
    col: Union[pd.Series, np.array], sample_size: Union[int, float] = 0.1
) -> np.array:
    all_idx = np.arange(len(col))
    if sample_size < 1:
        sample_size = int(len(col) * sample_size)
    return np.random.choice(all_idx, size=int(sample_size), replace=False)


def test_idx_filter(data_path: str, docs_size: int) -> np.array:
    """ Loads the index for train to avoid leakage """
    index_path = Path(data_path).parent / "tweeteval_test.csv"
    test_idx = pd.read_csv(index_path, index_col=0).index.values
    return ~np.isin(np.arange(docs_size), test_idx)


def write_pickle(obj: Any, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(args):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logging.info("loadin data...")
    all_docs = load_docs(args.data_path)
    all_embeddings = load_embeds(args.embedding_path)
    test_filter = test_idx_filter(args.data_path, docs_size=all_docs.shape[0])
    docs = all_docs[test_filter]
    embeddings = all_embeddings[test_filter, :]
    na_filter = ~docs.isna()
    docs = docs[na_filter].values
    embeddings = embeddings[na_filter, :]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.random.seed(0)
    sample_idx = get_random_idx(docs, sample_size=args.data_size)
    small_docs = docs[sample_idx]
    small_embs = embeddings[sample_idx, :]
    logging.debug("shape of data: %s", small_embs.shape)
    logging.info("bootin model...")
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english",)
    umap_model = UMAP(n_neighbors=15, n_components=50, min_dist=0.0)

    topic_model = BERTopic(
        low_memory=True,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=False,
        nr_topics=10,
    )
    logging.info("fittin model...")
    topics, probs = topic_model.fit_transform(small_docs, small_embs)
    logging.info("savin model...")
    topic_model.save(
        str((Path(args.save_path) / f"{time_stamp}_topic_model")),
        save_embedding_model=False,
    )
    logging.info("write topics")
    write_pickle(topic_model.get_topics(), Path("../data/tweeteval_topic_dict.pkl"))
    logging.info("done!")
    logging.info("all done!")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Embed Documents")
    my_parser.add_argument(
        "--data-path", type=str, help="Gives the path to the data file (a csv)"
    )
    my_parser.add_argument(
        "--embedding-path", type=str, help="Gives the path to the embeddings (.npy)",
    )
    my_parser.add_argument(
        "--save-path", type=str, help="Path to directory to save stuff",
    )
    my_parser.add_argument(
        "--data-size",
        type=float,
        help="Size of the data to use (100k seems to be a hard limit)",
    )
    args = my_parser.parse_args()
    main(args)

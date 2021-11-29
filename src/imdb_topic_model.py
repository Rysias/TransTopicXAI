from typing import Dict, List, Union
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer


def get_model_name(file_path: Path) -> str:
    full_name = file_path.name
    stop_idx = full_name.find("_")
    return full_name[:stop_idx]


def get_train_filter(train_index: pd.Index) -> np.ndarray:
    return np.isin(np.arange(50000), train_index)


# Paths
DATA_DIR = Path("../data")
MODEL_PATH = Path("../models")

if __name__ == "__main__":
    embedding_paths = list(DATA_DIR.glob("*embs.npy"))

    # Load data
    train_data = pd.read_csv(DATA_DIR / "imdb_train.csv", index_col=0)
    train_filter = get_train_filter(train_data.index)

    # Create vectorizer
    # Looping over the models
    for emb_path in embedding_paths:
        emb_path = embedding_paths[0]  # REMOVE THIS
        model_name = get_model_name(emb_path)
        print(f"processing {model_name}")

        all_embs = np.load(DATA_DIR / emb_path)
        embeddings = all_embs[train_filter, :]

        # Fitting models
        print("fitting model...")
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        nr_topics = 10
        hdbscan_model = hdbscan.HDBSCAN(
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True,
        )
        umap_model = UMAP(
            n_neighbors=5,
            n_components=50,
            min_dist=0.0,
            metric="cosine",
            low_memory=True,
        )
        topic_model = BERTopic(
            language="english",
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            # hdbscan_model=hdbscan_model,
            verbose=True,
            calculate_probabilities=True,
            nr_topics=nr_topics,
        )
        topics, probs = topic_model.fit_transform(train_data["text"], embeddings)
        # Saving predictions
        print("saving predictions...")
        np.save(DATA_DIR / f"{model_name}_probs.npy", probs)

        #        preds_df = pd.DataFrame(
        #            list(zip(topics, probs, train_data["text"])),
        #            columns=["topic", "prob", "doc"],
        #        )
        #        preds_df.to_csv(DATA_DIR / f"{model_name}_doc_topics.csv", index=False)
        print("saving model")
        topic_model.save(str(MODEL_PATH / f"{model_name}_topic_model"))
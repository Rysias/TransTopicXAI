from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def get_model_name(file_path: Path) -> str:
    full_name = file_path.name
    stop_idx = full_name.find("_")
    return full_name[:stop_idx]


# Paths
DATA_DIR = Path("../data")
MODEL_PATH = Path("../models")

if __name__ == "__main__":
    embedding_paths = list(DATA_DIR.glob("*embs.npy"))

    # Load data
    train_data = pd.read_csv(DATA_DIR / "imdb_train.csv")
    ids = np.load(DATA_DIR / "all_ids.npy", allow_pickle=True)

    train_data.set_index(train_data["id"], inplace=True)
    id_df = pd.DataFrame(ids, columns=["id"]).set_index("id")

    train_data.index.unique()
    id_df.join(train_data, how="inner")
    # Create vectorizer
    # Looping over the models
    for emb_path in embedding_paths:
        model_name = get_model_name(emb_path)
        print(f"processing {model_name}")

        embeddings = np.load(DATA_DIR / emb_path)
        train_mask = np.isin(ids, train_data["id"].values)

        # Fitting models
        print("fitting model...")
        topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=10)
        topics, probs = topic_model.fit_transform(full_docs, full_embs)

        # Saving predictions
        print("saving predictions...")
        preds_df = pd.DataFrame(
            list(zip(topics, probs, full_docs)), columns=["topic", "prob", "doc"]
        )
        preds_df.to_csv(DATA_DIR / f"{model_name}_doc_topics.csv", index=False)

        print("saving model")
        topic_model.save(str(MODEL_PATH / f"{model_name}_topic_model"))

        # Saving topics
        topic_dict = topic_model.get_topics()
        topic_dict_clean = {
            k: [tup[0] for tup in word_list]
            for k, word_list in topic_dict.items()
            if k != -1
        }
        pickle_object(topic_dict_clean, DATA_DIR / f"{model_name}_topic_dict.pkl")

import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def pickle_object(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def flatten_embeddings(embedding_dict):
    """Creates a big matrix with all the embeddings from the dict"""
    return np.vstack(embedding_dict.values())


def flatten_list(lst):
    return [elem for sublist in lst for elem in sublist]


def get_paragraphs(paragraph_dict):
    return flatten_list(list(paragraph_dict.values()))


def load_text_url(text_url: str):
    """Inputs a URL of a newline seperated text-file and returns a list"""
    response = requests.get(text_url)
    return response.text.split("\n")


def get_model_name(file_path: Path) -> str:
    full_name = file_path.name
    stop_idx = full_name.find("_")
    return full_name[:stop_idx]


# Paths
DATA_DIR = Path("../../BscThesisData/data")
MODEL_PATH = Path("../models")
embedding_paths = list(DATA_DIR.glob("*_embedding_dict.pkl"))

# Load texts
clean_paragraphs = read_pickle(DATA_DIR / "paragraph_dict.pkl")
docs = get_paragraphs(clean_paragraphs)

# Creating vectorizer
STOP_WORD_URL = "https://gist.githubusercontent.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b/raw/305d8e3930cc419e909d49d4b489c9773f75b2d6/stopord.txt"
STOP_WORDS = load_text_url(STOP_WORD_URL)
vectorizer_model = CountVectorizer(stop_words=STOP_WORDS)
pickle_object(vectorizer_model, MODEL_PATH / "vectorizer.pkl")


# Looping over the models
for emb_path in embedding_paths:
    model_name = get_model_name(emb_path)
    print(f"processing {model_name}")
    embedding_dict = read_pickle(emb_path)

    embeddings = flatten_embeddings(embedding_dict)

    # Fitting models
    print("fitting model...")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=5)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Saving predictions
    print("saving predictions...")
    preds_df = pd.DataFrame(
        list(zip(topics, probs, docs)), columns=["topic", "prob", "doc"]
    )
    preds_df.to_csv(DATA_DIR / f"{model_name}_doc_topics.csv", index=False)

    # Saving topics
    topic_dict = topic_model.get_topics()
    topic_dict_clean = {
        k: [tup[0] for tup in word_list]
        for k, word_list in topic_dict.items()
        if k != -1
    }
    pickle_object(topic_dict_clean, DATA_DIR / f"{model_name}_topic_dict.pkl")

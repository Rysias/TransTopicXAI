"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
import numpy as np
import pickle
import requests
from spacy import load
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List


def serialize_data(dat, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dat, f)


def flatten_list(lst: List):
    return [elem for sublist in lst for elem in sublist]


def read_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_text_url(text_url: str): 
    """Inputs a URL of a newline seperated text-file and returns a list"""
    response = requests.get(text_url)
    return response.text.split("\n")

    
def remove_stopwords(doc, stop_words):
    """ Removes the stopwords from a spacy doc given a list of stopwords"""
    return " ".join([word.text for word in doc if not word.text.lower() in stop_words])


DATA_DIR = Path("../BscThesisData/data")
data_path = DATA_DIR / "paragraph_dict.pkl"
resume_dict = read_pickle(data_path)

# Powering up the transformer!
sentence_model = SentenceTransformer(
    "Maltehb/-l-ctra-danish-electra-small-cased")

embedding_dimension = sentence_model.encode("hejsa").shape[0]


# Preprocess embeddings
all_paragraphs = flatten_list(list(resume_dict.values()))
stop_word_url = "https://gist.githubusercontent.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b/raw/305d8e3930cc419e909d49d4b489c9773f75b2d6/stopord.txt"
stop_words = load_text_url(stop_word_url)

danish_nlp = load("da_core_news_sm")

tokenized_paragraphs = danish_nlp.tokenizer.pipe(all_paragraphs)
clean_paragraphs = [remove_stopwords(doc, stop_words) for doc in tokenized_paragraphs]
# Embedding the documents #
all_embeds = sentence_model.encode(all_paragraphs, show_progress_bar=True)

# Add embeddings to dictionary #
# Creating the output dictionary structure
embedding_dict = {doc_id: np.zeros((len(paragraphs), embedding_dimension))
                  for doc_id, paragraphs in resume_dict.items()}

current_idx = 0
for doc_id, emb_array in embedding_dict.items():
    stop_idx = current_idx + emb_array.shape[0]  # Makes it stop at the right number of docs
    embedding_dict[doc_id] += all_embeds[current_idx:stop_idx, :]
    current_idx += emb_array.shape[0]  # Reset to next document

# Writing data to disk
serialize_data(embedding_dict, DATA_DIR / "embedding_dict.pkl")

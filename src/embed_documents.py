"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer


def read_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


DATA_DIR = Path("../BscThesisData/data")
data_path = DATA_DIR / "paragraph_dict.pkl"
resume_dict = read_pickle(data_path)

# Powering up the transformer!
sentence_model = SentenceTransformer(
    "Maltehb/-l-ctra-danish-electra-small-cased")

embedding_dimension = sentence_model.encode("hejsa").shape[0]

# Creating the output dictionary
embedding_dict = {doc_id: np.zeros((len(paragraphs), embedding_dimension))
                  for doc_id, paragraphs in resume_dict.items()}


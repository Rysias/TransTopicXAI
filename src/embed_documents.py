"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
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
sentence_model = SentenceTransformer("Maltehb/-l-ctra-danish-electra-small-cased")

embedding_shape = sentence_model.encode("hejsa").shape
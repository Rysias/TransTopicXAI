from typing import Dict, List
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


DATA_DIR = Path("../../BscThesisData/data")
MODEL_PATH = Path("../models")


def read_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_embedding_df(embedding_dict: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    doc_embeddings = {k: emb.mean(axis=0) for k, emb in embedding_dict.items()}
    return pd.DataFrame.from_dict(doc_embeddings, orient="index")


label_df = pd.read_csv(DATA_DIR / "emne_labels.csv")
embedding_paths = list(DATA_DIR.glob("*_embedding_dict.pkl"))
embedding_dict = read_pickle(embedding_paths[0])
embedding_df = create_embedding_df(embedding_dict)

label_df.set_index("case_id", inplace=True)

label_df.join(embedding_df).dropna()

testy = {k: v for k, v in embedding_dict.items() if k in set(label_df["case_id"])}

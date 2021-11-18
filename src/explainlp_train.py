import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path("../../BscThesisData/data")
MODEL_PATH = Path("../models")


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


label_df = pd.read_csv(DATA_DIR / "emne_labels.csv")

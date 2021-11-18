import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../BscThesisData/data")
MODEL_PATH = Path("../models")

full_df = pd.read_csv(DATA_DIR / "full_df.csv")
full_df.head()

full_df.groupby("case_id").first("label")

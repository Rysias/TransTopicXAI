import pandas as pd
from explainlp.explainlp import ClearSearch
from pathlib import Path

model = ClearSearch(
    "Maltehb/-l-ctra-danish-electra-small-cased",
    topic_model=Path("../TransTopicXAI/models/topic_model"),
)


DATA_DIR = Path("../../BscThesisData/data")
doc_topics = pd.read_csv(DATA_DIR / "doc_topics.csv")

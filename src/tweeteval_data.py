import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union

DATA_DIR = Path("../tweeteval/datasets")


def scan_sentiment(file: Path) -> List[Union[None, int]]:
    if file.parent.name == "sentiment":
        return read_file(file)
    return [None for _ in read_file(file)]


def scan_type(file: Path) -> List[str]:
    set_types = ["train", "test", "val"]
    for typ in set_types:
        if file.name.startswith(typ):
            return [typ for _ in read_file(file)]
    return [None for _ in read_file(file)]


def read_file(file: Path) -> List[str]:
    with open(file, "r", encoding="utf8") as f:
        return f.readlines()


all_text = []
all_types = []
for file in DATA_DIR.rglob("*_text.txt"):
    all_text.extend(read_file(file))
    all_types.extend(scan_type(file))


all_labels = []
for file in DATA_DIR.rglob("*_labels.txt"):
    all_labels.extend(scan_sentiment(file))

set(all_labels)

df = pd.DataFrame({"text": all_text, "label": all_labels, "type": all_types})
df = df.assign(
    text=df["text"].str.strip(), label=df["label"].str.extract("(\d)").astype(float)
)

df["text"].to_csv(Path("data/tweeteval_text.csv"))


sent_df = df.dropna()

test_df = sent_df[sent_df["type"] == "test"].drop(columns="type")
test_df.to_csv(Path("data/tweeteval_test.csv"))
train_df = sent_df[sent_df["type"] != "test"].drop(columns="type")
train_df.to_csv(Path("data/tweeteval_train.csv"))

import pandas as pd
import pickle
from pprint import pprint
from pathlib import Path
from typing import Dict, Tuple, List

DATA_DIR = Path("../data")


def create_topic_dict(
    raw_topic_dict: Dict[int, Tuple[str, int]]
) -> Dict[int, List[str]]:
    return {k: [tup[0] for tup in lst] for k, lst in raw_topic_dict.items() if k != -1}


def latest_full_topics(dr: Path) -> Path:
    return sorted(dr.glob("*full_doc_topics_*.csv"))[-1]


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


doc_topics = pd.read_csv(latest_full_topics(DATA_DIR), index_col=0)
clean_tweets = pd.read_csv(DATA_DIR / "tweeteval_text.csv", usecols=["text"])
doc_topics = doc_topics.merge(clean_tweets, left_index=True, right_index=True)
topic_words = read_pickle(DATA_DIR / "tweeteval_topic_dict.pkl")
topic_dict = create_topic_dict(topic_words)
for top, words in topic_dict.items():
    print(f"evaluating topic {top}")
    topic_tweets = doc_topics.loc[doc_topics["topic"] == top, "text"]
    print(f"num tweets in {top}: {topic_tweets.shape[0]}")
    pprint(f"{words = }")
    print(f"examples for topic {top}")
    example_tweets = topic_tweets.sample(10, random_state=42).tolist()
    print(example_tweets)
    print("")

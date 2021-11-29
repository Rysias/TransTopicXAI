import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_DIR = Path("../ExplainlpTwitter/output/")
OUTPUT_DIR = Path("data/")
all_tweets = pd.read_csv(DATA_DIR / "clean_tweets.csv")
train_tweets, test_tweets = train_test_split(
    all_tweets, test_size=10000, random_state=42
)


sample_tweets = test_tweets.sample(20, random_state=1)

tweety = sample_tweets[["cleantext", "Sentiment"]]
bert_test = tweety.sample(10, random_state=2)
explainlp_test = tweety.drop(index=bert_test.index)

bert_test.to_csv(OUTPUT_DIR / "bert_test.csv")
explainlp_test.to_csv(OUTPUT_DIR / "explainlp_test.csv")

test = pd.read_csv(OUTPUT_DIR / "explainlp_test.csv", index_col=0)

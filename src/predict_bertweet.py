import pandas as pd
import sys
from argparse import ArgumentParser
from pysentimiento import create_analyzer
from pathlib import Path
from pysentimiento.analyzer import AnalyzerOutput


def parse_prediction(pred: AnalyzerOutput) -> float:
    label_dict = {"NEG": 0.0, "NEU": 1.0, "POS": 2.0}
    return label_dict[pred.output]


if __name__ == "__main__":
    parser = ArgumentParser(description="predict with the bertweet")
    parser.add_argument("--data-dir", default="../../final_proj_dat")
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    pred_path = DATA_DIR / "bertweet_preds.csv"
    if pred_path.exists():
        print("already exists")
        sys.exit()
    test_df = pd.read_csv(DATA_DIR / "tweeteval_test.csv")
    sentiment = create_analyzer("sentiment", "en")
    preds = sentiment.predict(test_df["text"])
    parsed_preds = [parse_prediction(pred) for pred in preds]
    bertweet_preds = pd.DataFrame({"y_true": test_df["label"], "y_pred": parsed_preds})
    bertweet_preds.to_csv(pred_path)

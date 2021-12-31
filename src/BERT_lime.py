"""
Creates LIME graphs + raw predictions
"""
from typing import Dict, List, Union
import pandas as pd
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from pysentimiento import create_analyzer
from lime.lime_text import LimeTextExplainer
from pysentimiento.analyzer import AnalyzerOutput


def sort_sentiment(res: AnalyzerOutput) -> np.array:
    vals = [res.probas[k] for k in LABELS]
    return np.array(vals).reshape(1, -1)


def list_to_arr(result: List[AnalyzerOutput]) -> np.ndarray:
    return np.vstack([sort_sentiment(out) for out in result])


def format_output(result: Union[List[AnalyzerOutput], AnalyzerOutput]) -> np.ndarray:
    try:
        return sort_sentiment(result)
    except AttributeError:
        return list_to_arr(result)


def dict_to_arr(dct: dict) -> np.ndarray:
    n_feats = len(dct.values())
    return np.array(list(dct.values())).reshape((-1, n_feats))


def predict_proba(sentence: str) -> np.ndarray:
    pred = SENTIMENT.predict(sentence)
    return format_output(pred)


def get_pred_idx(sentence: str) -> int:
    return np.argmax(predict_proba(sentence))


def pred_loop(df: pd.DataFrame, explainer: LimeTextExplainer):
    for i, row in df.iterrows():
        pred_idx = get_pred_idx(row["text"])
        explanation = explainer.explain_instance(
            row["text"], predict_proba, num_features=5, labels=[pred_idx],
        )
        explanation.as_pyplot_figure(label=pred_idx)
        plt.savefig(DATA_DIR / f"bertplot_{i}.png")


def write_preds(df: pd.DataFrame):
    preds = SENTIMENT.predict(df["text"].tolist())
    pred_list = [{pred.output: max(pred.probas.values())} for pred in preds]
    for tweet_id, pred_dict in zip(df.index, pred_list):
        with open(DATA_DIR / f"bertpred_{tweet_id}.json", "w") as f:
            json.dump(pred_dict, f)


def main():
    explainer = LimeTextExplainer(class_names=LABELS)
    # Test on data
    df = pd.read_csv(DATA_DIR / "bert_test.csv", index_col=0)

    # Create txts with predictions
    write_preds(df)

    # Pred and Plot stuff
    pred_loop(df, explainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create BERT explanation PNGs")
    parser.add_argument("--data-dir", type=str, help="Where to find and load the data")

    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    SENTIMENT = create_analyzer("sentiment", lang="en")
    LABELS = list(SENTIMENT.id2label.values())
    main()

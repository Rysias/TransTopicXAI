from typing import Dict, List, Union
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import dill
from pathlib import Path
from pysentimiento import create_analyzer
from lime.lime_text import LimeTextExplainer
from pysentimiento.analyzer import AnalyzerOutput

DATA_DIR = Path("data/")


def dump_dill(obj, file_path: Path):
    with open(file_path, "wb") as f:
        dill.dump(obj, f)


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
        pred_idx = get_pred_idx(row["cleantext"])
        explanation = explainer.explain_instance(
            row["cleantext"],
            predict_proba,
            num_features=5,
            labels=[pred_idx],
        )
        dump_dill(explanation, DATA_DIR / f"explanation_{i}.pkl")


def main():
    explainer = LimeTextExplainer(class_names=LABELS)
    # Test on data
    df = pd.read_csv(DATA_DIR / "bert_test.csv", index_col=0)
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

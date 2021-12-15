from datetime import datetime
from typing import Dict, Union, Callable
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from sklearn import metrics
from sentence_transformers import SentenceTransformer
from functools import partial
from numpy.typing import ArrayLike

OUTPUT_DIR = Path("data")


def f1_pos_neg(y_true: ArrayLike, y_pred: ArrayLike):
    """ Calculates the average f1 score of positive and negative predictions """
    return np.mean(metrics.f1_score(y_true, y_pred, average=None)[[0, 2]])


def evaluate_preds(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    eval_dict: Dict[str, Callable],
) -> Dict[str, float]:
    return {metric: func(y_true, y_pred) for metric, func in eval_dict.items()}


def get_latest_preds(pattern: str) -> pd.DataFrame:
    return sorted(OUTPUT_DIR.glob(f"*{pattern}_preds*.csv"))[-1]


EVAL_DICT = {
    "Accuracy": metrics.accuracy_score,
    "F1_PN": f1_pos_neg,
    # "precision": partial(metrics.precision_score, average="macro"),
    "AvgRec": partial(metrics.recall_score, average="macro"),
}
# Get time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Evaluating my model
latest_pred_path = get_latest_preds("topic")
topic_preds = pd.read_csv(latest_pred_path)

# Evaluating tf-idf
latest_tfidf = get_latest_preds("tf_idf")
tfidf_preds = pd.read_csv(latest_tfidf)
# Evaluating sentiment on same data!
big_preds = pd.read_csv(OUTPUT_DIR / "bertweet_preds.csv")

print("Evaluation of topic model")
topic_results = evaluate_preds(topic_preds["y_true"], topic_preds["y_pred"], EVAL_DICT)
print("Evaluation of big model")
bert_results = evaluate_preds(big_preds["y_true"], big_preds["y_pred"], EVAL_DICT)
print("Evaluation of tfidf")
tfidf_results = evaluate_preds(tfidf_preds["y_true"], tfidf_preds["y_pred"], EVAL_DICT)


# Count number of parameters
model = SentenceTransformer("finiteautomata/bertweet-base-sentiment-analysis")
total_num = sum(p.numel() for p in model.parameters())

# Saving results
bert_df = pd.DataFrame(bert_results, index=[0]).assign(
    model="transentiment", num_params=total_num
)
topic_df = pd.DataFrame(topic_results, index=[1]).assign(
    model="topic-based", num_params=10
)
tfidf_df = pd.DataFrame(tfidf_results, index=[2]).assign(
    model="tf-idf", num_params=2000
)
all_results = pd.concat((bert_df, topic_df, tfidf_df))
all_results.to_csv(OUTPUT_DIR / f"comparison_results_{current_time}.csv")


semeval_results = """
1 DataStories 0.6811 0.6772 0.6515
1 BB_twtr 0.6811 0.6851 0.6583
3 LIA 0.6763 0.6743 0.6612
4 Senti17 0.6744 0.6654 0.6524
5 NNEMBs 0.6695 0.6585 0.6641
6 Tweester 0.6596 0.6486 0.6486
7 INGEOTEC 0.6497 0.6457 0.63311
8 SiTAKA 0.6458 0.6289 0.6439
"""

semeval = pd.read_csv(
    StringIO(semeval_results),
    sep="\s",
    header=None,
    names=["rank", "model", "AvgRec", "F1_PN", "Accuracy"],
)
semeval.drop(columns="rank", inplace=True)

semeval.append(all_results[semeval.columns]).sort_values(by="AvgRec", ascending=False)

semeval

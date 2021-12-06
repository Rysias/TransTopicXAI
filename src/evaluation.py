from datetime import datetime
from typing import Dict, Union, Callable
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import metrics
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("data")
EVAL_DICT = {
    "accuracy": metrics.accuracy_score,
    "f1": metrics.f1_score,
    "roc_auc": metrics.roc_auc_score,
    "precision": metrics.precision_score,
    "recall": metrics.recall_score,
}


def evaluate_preds(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    eval_dict: Dict[str, Callable],
) -> Dict[str, float]:
    return {metric: func(y_true, y_pred) for metric, func in eval_dict.items()}


# Get time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Evaluating my model
latest_pred_path = list(OUTPUT_DIR.glob("*topic_preds_*.csv"))[-1]
topic_preds = pd.read_csv(latest_pred_path)

# Evaluating sentiment on same data!
big_preds = pd.read_csv(OUTPUT_DIR / "big_preds.csv")
big_preds["id"] = big_preds["id"].astype(np.uint64)
small_preds = big_preds[big_preds["id"].isin(topic_preds["id"])]
y_true = small_preds["Sentiment"]
y_pred = small_preds["pred"]

print("Evaluation of topic model")
topic_results = evaluate_preds(topic_preds["y_true"], topic_preds["y_pred"], EVAL_DICT)
print("Evaluation of big model")
bert_results = evaluate_preds(y_true, y_pred, EVAL_DICT)


# Count number of parameters
model = SentenceTransformer("finiteautomata/bertweet-base-sentiment-analysis")
total_num = sum(p.numel() for p in model.parameters())

# Saving results
bert_df = pd.DataFrame(bert_results, index=[0]).assign(
    model="bertweet", num_params=total_num
)
topic_df = pd.DataFrame(topic_results, index=[1]).assign(model="topic", num_params=10)
all_results = pd.concat((bert_df, topic_df,))
all_results.to_csv(OUTPUT_DIR / f"comparison_results_{current_time}.csv")

from datetime import datetime
from typing import Dict, Union, Callable
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


EVAL_DICT = {
    "accuracy": metrics.balanced_accuracy_score,
    "f1pn": f1_pos_neg,
    "precision": partial(metrics.precision_score, average="macro"),
    "recall": partial(metrics.recall_score, average="macro"),
}
# Get time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Evaluating my model
latest_pred_path = sorted(OUTPUT_DIR.glob("*topic_preds_*.csv"))[-1]
topic_preds = pd.read_csv(latest_pred_path)

# Evaluating sentiment on same data!
big_preds = pd.read_csv(OUTPUT_DIR / "bertweet_preds.csv")

print("Evaluation of topic model")
topic_results = evaluate_preds(topic_preds["y_true"], topic_preds["y_pred"], EVAL_DICT)
print("Evaluation of big model")
bert_results = evaluate_preds(big_preds["y_true"], big_preds["y_pred"], EVAL_DICT)


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


# Plotting #
sns.set_theme(style="whitegrid")
plot_results = pd.melt(
    all_results, id_vars=["model", "num_params"], var_name="metric", value_name="score"
)
sns.catplot(data=plot_results, kind="bar", x="metric", y="score", hue="model").set(
    title="Bertweet vs topic model"
)
plt.savefig(Path("data/result_graph.png"))

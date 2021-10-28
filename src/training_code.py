import pandas as pd
import numpy as np
from sklearn import metrics
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from typing import Dict, Generator, Tuple


def create_k_folds(
    train_df: pd.DataFrame, k=3
) -> Generator[None, Tuple[pd.DataFrame, pd.DataFrame], None]:
    group_kfold = GroupKFold(n_splits=k)
    for train_idx, test_idx in group_kfold.split(
        train_df.drop("case_id", axis=1),
        train_df["label"],
        train_df["case_id"],
    ):
        yield train_idx, test_idx


def create_label_dict(label_col: pd.Series) -> Dict[str, int]:
    labels = label_col.unique()
    return {label: i for i, label in enumerate(labels)}


all_data = pd.read_csv(Path("../../BscThesisData/data/full_df.csv"))
unique_ids = all_data["case_id"].unique()
train_ids, test_ids = train_test_split(unique_ids, random_state=42, train_size=0.8)

training_data = all_data.loc[np.isin(all_data["case_id"], train_ids)].reset_index(
    drop=True
)

label_dict = create_label_dict(all_data["label"])
training_data = training_data.replace({"label": label_dict})
num_labels = len(label_dict)


model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
model_type = "electra"
model_args = ClassificationArgs(num_train_epochs=1, overwrite_output_dir=True)
results = []
for train_id, val_id in create_k_folds(training_data):
    train_dat = training_data.loc[train_id, ["resume", "label"]]
    eval_dat = training_data.loc[val_id, ["resume", "label"]]

    model = ClassificationModel(
        model_type, model_name, num_labels=num_labels, args=model_args, use_cuda=True
    )
    model.train_model(train_dat)
    result, model_outputs, wrong_predictions = model.eval_model(
        eval_dat, acc=metrics.f1_score
    )
    results.append(result["acc"])

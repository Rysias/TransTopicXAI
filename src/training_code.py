import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from typing import Generator, Tuple


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


all_data = pd.read_csv(Path("../../BscThesisData/data/full_df.csv"))
unique_ids = all_data["case_id"].unique()
train_ids, test_ids = train_test_split(unique_ids, random_state=42, train_size=0.8)

training_data = all_data.loc[np.isin(all_data["case_id"], train_ids)].reset_index(
    drop=True
)
labels = training_data["label"].unique()
label_dict = {label: i for i, label in enumerate(labels)}
training_data = training_data.replace({"label": label_dict})

for train_id, val_id in create_k_folds(training_data):
    print(training_data.loc[train_id, :])
    break


num_labels = len(labels)


model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
model_type = "electra"
model_args = ClassificationArgs(num_train_epochs=1, overwrite_output_dir=True)

model = ClassificationModel(
    model_type, model_name, num_labels=num_labels, args=model_args, use_cuda=False
)

model.train_model(training_data, overwrite_output_dir=True)

import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split, GroupShuffleSplit


all_data = pd.read_csv(Path("../../BscThesisData/data/full_df.csv"))
unique_ids = all_data["case_id"].unique()
train_ids, test_ids = train_test_split(unique_ids, random_state=42, train_size=0.8)

training_data = all_data.loc[np.isin(all_data["case_id"], train_ids)]
labels = training_data["label"].unique()
label_dict = {label: i for i, label in enumerate(labels)}
training_data = training_data.replace({"label": label_dict})

training_data.drop("case_id", axis=1)
group_kfold = GroupKFold(n_splits=3)
for train_idx, test_idx in group_kfold.split(
    training_data.drop("case_id", axis=1),
    training_data["label"],
    training_data["case_id"],
):
    print(train_idx)
    break


num_labels = len(labels)


model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
model_type = "electra"
model_args = ClassificationArgs(num_train_epochs=1, overwrite_output_dir=True)

model = ClassificationModel(
    model_type, model_name, num_labels=num_labels, args=model_args, use_cuda=False
)

model.train_model(training_data, overwrite_output_dir=True)

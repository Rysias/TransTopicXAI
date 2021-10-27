import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pathlib import Path

all_data = pd.read_csv(Path("../../BscThesisData/data/full_df.csv"))
training_data = all_data.sample(100)[["resume", "label"]]

labels = training_data["label"].unique()
label_dict = {label: i for i, label in enumerate(labels)}
training_data = training_data.replace({"label": label_dict})


num_labels = len(labels)


model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
model_type = "electra"
model_args = ClassificationArgs(num_train_epochs=1)

model = ClassificationModel(
    model_type, model_name, num_labels=num_labels, args=model_args, use_cuda=False
)

model.train_model(training_data)

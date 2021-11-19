import pandas as pd
import numpy as np
import argparse
import pickle
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Generator, Tuple


def create_label_dict(label_col: pd.Series) -> Dict[str, int]:
    labels = label_col.unique()
    return {label: i for i, label in enumerate(labels)}


def write_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(args):
    all_data = pd.read_csv(Path(args.data_path))
    label_dict = create_label_dict(all_data["label"])
    all_data = all_data.replace({"label": label_dict})
    unique_ids = all_data["case_id"].unique()
    train_ids, test_ids = train_test_split(unique_ids, random_state=42, train_size=0.8)

    training_data = all_data.loc[np.isin(all_data["case_id"], train_ids)].reset_index(
        drop=True
    )

    num_labels = len(label_dict)
    training_data = training_data[["resume", "label"]]

    model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
    model_type = "electra"
    model_args = ClassificationArgs(num_train_epochs=1, sliding_window=True)
    model = ClassificationModel(
        model_type, model_name, num_labels=num_labels, args=model_args, use_cuda=True
    )
    model.train_model(training_data, output_dir=str(Path(f"./{model_type}_outputs")))

    test_data = all_data.loc[np.isin(all_data["case_id"], test_ids)].reset_index(
        drop=True
    )
    predictions = model.predict(test_data["resume"])
    write_pickle(predictions, Path(args.save_path) / f"{model_type}_preds.pkl")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="train model")
    my_parser.add_argument(
        "--data_path", type=str, help="Gives the path to the data file (a csv)"
    )
    my_parser.add_argument("--save_path", type=str, help="gives path to save output")
    args = my_parser.parse_args()
    main(args)

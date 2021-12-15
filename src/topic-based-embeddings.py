from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from bertopic import BERTopic
from explainlp.clearformer import Clearformer


def load_latest_topic_model(dr: Path):
    topic_path = sorted(dr.glob(f"*topic_model"))[-1]
    return BERTopic.load(topic_path)


def load_embeddings(dr: Path) -> np.ndarray:
    return np.load(dr / "bertweet-base-sentiment-analysis_embs.npy")


def load_target(dr: Path, settype="train"):
    target_path = dr / f"tweeteval_{settype}.csv"
    return pd.read_csv(target_path)["label"]


def main(args):
    current_time = datetime.now().strftime("%y%m%d%H")
    DATA_DIR = Path(args.data_dir)

    all_embs = load_embeddings(DATA_DIR)
    Y_train = load_target(DATA_DIR, settype="train")
    Y_test = load_target(DATA_DIR, settype="test")
    X_train = all_embs[Y_train.index.values, :]
    X_test = all_embs[Y_test.index.values, :]

    # initializing topic model
    topic_model = load_latest_topic_model(Path(args.topic_dir))
    clearformer = Clearformer(topic_model)

    normalizer = MinMaxScaler()
    poly = PolynomialFeatures(interaction_only=True, include_bias=True)
    model = LogisticRegression(solver="saga", penalty="l1")
    grid = {"logistic__C": np.logspace(-1, 5, 10)}  # l1 lasso
    pipeline = Pipeline(
        steps=[
            ("topic-embed", clearformer),
            ("poly", poly),
            ("normalize", normalizer),
            ("logistic", model),
        ],
        verbose=True,
    )
    gridsearch = GridSearchCV(pipeline, param_grid=grid, cv=3, verbose=True, n_jobs=-1)
    gridsearch.fit(X_train, Y_train)
    y_preds = gridsearch.predict(X_test)
    pd.DataFrame({"y_true": Y_test, "y_pred": y_preds}).to_csv(
        DATA_DIR / f"topic_preds_{current_time}.csv", index=False
    )

    joblib.dump(
        gridsearch.best_estimator_, DATA_DIR / f"topic_predictor_{current_time}.joblib"
    )


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="train topic model sentiment stuff")
    my_parser.add_argument(
        "--data-dir",
        type=str,
        help="Gives the path to the data directory",
        default="../../final_proj_dat",
    )
    my_parser.add_argument(
        "--topic-dir",
        type=str,
        help="Gives the dir to the topic model",
        default="../../final_proj_dat/",
    )
    args = my_parser.parse_args()
    main(args)

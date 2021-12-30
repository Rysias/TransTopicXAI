"""
Fits the topic-based classifier (including creating topic embeddings)
"""
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import logging
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

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    current_time = datetime.now().strftime("%y%m%d%H")
    DATA_DIR = Path(args.data_dir)
    logging.info("Loading data...")
    all_embs = load_embeddings(DATA_DIR)
    Y_train = load_target(DATA_DIR, settype="train")
    Y_test = load_target(DATA_DIR, settype="test")
    X_train_raw = all_embs[Y_train.index.values, :]
    logging.debug(f"{X_train_raw.shape = }")
    X_test_raw = all_embs[Y_test.index.values, :]
    logging.info("Finished loading data!")

    # initializing topic model
    logging.info("initializing topic modelling....")
    topic_model = load_latest_topic_model(Path(args.topic_dir))
    clearformer = Clearformer(topic_model)

    # creating full_docs stuff
    logging.info("writing topics to file")
    full_doc_topic_path = DATA_DIR / f"full_doc_topics_{current_time}.csv"
    random_docs = [
        "i" for _ in range(X_train_raw.shape[0])
    ]  # Stupid hack as bertopic doesn't actually use documents for transform but requires it as an argument
    topics, _ = topic_model.transform(random_docs, X_train_raw)
    full_doc_topic = pd.DataFrame({"topic": topics}, index=Y_train.index)
    full_doc_topic.to_csv(full_doc_topic_path)
    logging.info("done")

    # Creating topic embeddings
    X_train = clearformer.fit_transform(X_train_raw)
    X_test = clearformer.transform(X_test_raw)
    logging.info("Done with topic embeddings!")

    logging.info("Commencing training...")
    normalizer = MinMaxScaler()
    # poly = PolynomialFeatures(interaction_only=True, include_bias=True)
    model = LogisticRegression(solver="saga", penalty="l1")
    grid = {"logistic__C": np.logspace(-1, 5, 10)}  # l1 lasso
    pipeline = Pipeline(
        steps=[
            # ("poly", poly),
            ("normalize", normalizer),
            ("logistic", model),
        ],
        verbose=True,
    )
    gridsearch = GridSearchCV(pipeline, param_grid=grid, cv=3, verbose=True, n_jobs=-1)
    gridsearch.fit(X_train, Y_train)
    logging.info("done!")
    logging.info("saving results...")
    y_preds = gridsearch.predict(X_test)
    pd.DataFrame({"y_true": Y_test, "y_pred": y_preds}).to_csv(
        DATA_DIR / f"topic_preds_{current_time}.csv", index=False
    )

    joblib.dump(
        gridsearch.best_estimator_, DATA_DIR / f"topic_predictor_{current_time}.joblib"
    )
    logging.info("All done!")


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

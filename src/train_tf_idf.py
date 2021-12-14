from typing import Tuple
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path


def load_feats_and_target(dr: Path, filename: str) -> Tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(dr / filename, index_col=0)
    return feats_and_target(df)


def feats_and_target(dataset: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
    return dataset["text"], dataset["label"].values


def main(args) -> None:
    current_time = datetime.now().strftime("%y%m%d%H")
    DATA_DIR = Path(args.data_dir)
    X_train, y_train = load_feats_and_target(DATA_DIR, "tweeteval_train.csv")
    X_test, y_test = load_feats_and_target(DATA_DIR, "tweeteval_test.csv")
    tfidf = TfidfVectorizer()
    model = LogisticRegression(solver="saga", penalty="l1")
    grid = {"logistic__C": np.logspace(-1, 5, 10)}  # l1 lasso

    pipeline = Pipeline(steps=[("tfidf", tfidf), ("logistic", model)])
    gridsearch = GridSearchCV(pipeline, param_grid=grid, cv=3, verbose=True, n_jobs=-1)

    gridsearch.fit(X_train, y_train)
    y_preds = gridsearch.predict(X_test)
    pd.DataFrame({"y_true": y_test, "y_pred": y_preds}).to_csv(
        DATA_DIR / f"tf_idf_preds_{current_time}.csv", index=False
    )

    joblib.dump(
        gridsearch.best_estimator_,
        Path(args.model_dir) / f"tf_idf_{current_time}.joblib",
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Trains a TF-IDF based logisic regression on the TweetEval dataset"
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory where to find the training and test-set",
    )
    parser.add_argument("--model-dir", help="Directory where to save the models")
    args = parser.parse_args()
    main(args)

from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


def main(args):
    current_time = datetime.now().strftime("%y%m%d%H")
    DATA_DIR = Path(args.data_dir)
    all_embs = np.load(DATA_DIR / "topic_embs.npy")
    train_df = pd.read_csv(DATA_DIR / "clean_tweets.csv")
    train_df.head()

    fullX = all_embs
    fullY = train_df["Sentiment"].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        fullX, fullY, test_size=10000, random_state=42
    )

    normalizer = MinMaxScaler()
    poly = PolynomialFeatures(interaction_only=True, include_bias=True)
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
        "--data-dir", type=str, help="Gives the path to the data directory"
    )
    args = my_parser.parse_args()
    main(args)

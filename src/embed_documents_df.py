"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional


def serialize_data(dat, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dat, f)


def flatten_list(lst: List):
    return [elem for sublist in lst for elem in sublist]


def read_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_emb_path(
    arg_dir: Union[str, Path, None], embedding_name: str
) -> Optional[Path]:
    if arg_dir is None:
        return None
    return Path(arg_dir) / embedding_name


def main(args):
    DATA_DIR = Path("../BscThesisData/data")
    data_path = (
        Path(args.data_path) if args.data_path else (DATA_DIR / "paragraph_dict.pkl")
    )
    print(f"loading data from {data_path}...")
    resume_df = pd.read_csv(data_path / "full_df.csv")
    print("loaded the data!")
    all_paragraphs = resume_df["resume"].values
    # Powering up the transformer!
    transformer_list = [
        "Maltehb/-l-ctra-danish-electra-small-cased",
        "Maltehb/danish-bert-botxo",
        "xlm-roberta-large",
    ]

    for transformer in transformer_list:
        print(f"ready for {transformer}!")
        temp_df = resume_df.copy()
        temp_df.drop(columns="resume", inplace=True)
        # Get the name of the transformer
        transformer_name = transformer.split("/")[-1]
        # Boot it up
        sentence_model = SentenceTransformer(transformer)

        # Embedding the documents #
        print("Embedding documents!")
        all_embeds = sentence_model.encode(all_paragraphs, show_progress_bar=True)
        temp_df["embeddings"] = all_embeds

        print("done with embedding...")
        # Writing data to disk
        embedding_name = f"{transformer_name}_emb_df.csv"
        embedding_path_local = DATA_DIR / embedding_name
        embedding_path_arg = create_emb_path(args.embedding_path, embedding_name)
        embedding_path = (
            embedding_path_arg if embedding_path_arg else embedding_path_local
        )
        print(f"written to disk at {embedding_path}!")
        temp_df.to_csv(embedding_path)

    print("all done!")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Embed Documents")
    my_parser.add_argument(
        "--data_path", type=str, help="Gives the path to the data file (a pickle)"
    )
    my_parser.add_argument(
        "-emb",
        "--embedding_path",
        type=str,
        required=False,
        help="Gives the directory where to save embeddings",
    )
    args = my_parser.parse_args()
    main(args)

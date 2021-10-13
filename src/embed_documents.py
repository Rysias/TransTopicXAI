"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
import numpy as np
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


def create_embedding_dict(resume_dict, embeddings, embedding_dimension):
    # Add embeddings to dictionary #
    # Creating the output dictionary structure
    embedding_dict = {
        doc_id: np.zeros((len(paragraphs), embedding_dimension))
        for doc_id, paragraphs in resume_dict.items()
    }

    current_idx = 0
    for doc_id, emb_array in embedding_dict.items():
        stop_idx = (
            current_idx + emb_array.shape[0]
        )  # Makes it stop at the right number of docs
        embedding_dict[doc_id] += embeddings[current_idx:stop_idx, :]
        current_idx += emb_array.shape[0]  # Reset to next document
    return embedding_dict


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
    resume_dict = read_pickle(data_path)
    print("loaded the data!")
    all_paragraphs = flatten_list(list(resume_dict.values()))
    # Powering up the transformer!
    transformer_list = [
        # "chcaa/da_dacy_small_trf",
        # "chcaa/da_dacy_medium_trf",
        # "chcaa/da_dacy_large_trf",
        "maltehb/-l-ctra-danish-electra-small-cased",
        "saattrupdan/nbailab-base-ner-scandi",
    ]

    for transformer in transformer_list:
        print(f"ready for {transformer}!")
        # Get the name of the transformer
        transformer_name = transformer.split("/")[1]
        # Boot it up
        sentence_model = SentenceTransformer(transformer)

        # Embedding the documents #
        print("Embedding documents!")
        all_embeds = sentence_model.encode(all_paragraphs, show_progress_bar=True)
        embedding_dimension = sentence_model.encode("en").shape[0]
        embedding_dict = create_embedding_dict(
            resume_dict, all_embeds, embedding_dimension
        )

        print("done with embedding...")
        # Writing data to disk
        embedding_name = f"{transformer_name}_embedding_dict.pkl"
        embedding_path_local = DATA_DIR / embedding_name
        embedding_path_arg = create_emb_path(args.embedding_path, embedding_name)
        embedding_path = (
            embedding_path_arg if embedding_path_arg else embedding_path_local
        )
        serialize_data(embedding_dict, embedding_path)
        print(f"written to disk at {embedding_path}!")
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

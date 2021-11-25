"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
import numpy as np
import pandas as pd
import pickle
import torch
import argparse
from tqdm import tqdm
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


def split_array(a: np.ndarray, chunk_size=1000) -> List[np.ndarray]:
    num_chunks = a.shape[0] // chunk_size + 1
    return np.array_split(a, num_chunks)


def process_large_model(
    text: np.ndarray, model: SentenceTransformer, chunk_size=500
) -> np.ndarray:
    emb_dims = model.encode("hej").shape[0]
    embeddings = np.zeros((text.shape[0], emb_dims))
    current_idx = 0
    for chunk in tqdm(split_array(text, chunk_size=chunk_size)):
        stop_idx = (
            current_idx + chunk.shape[0]
        )  # Makes it stop at the right number of docs
        new_embeddings = model.encode(chunk)
        embeddings[current_idx:stop_idx, :] += new_embeddings
        current_idx += chunk.shape[0]  # Reset to next document
        del new_embeddings
        torch.cuda.empty_cache()
    return embeddings


def main(args):
    data_path = Path(args.data_path)

    print(f"loading data from {data_path}...")
    text_df = pd.read_csv(data_path / "all_data.csv")
    print("loaded the data!")
    all_paragraphs = text_df["text"].values
    # Powering up the transformer!
    transformer_list = [
        "allenai/longformer-base-4096",
        "allenai/longformer-large-4096",
    ]

    for transformer in transformer_list:
        print(f"ready for {transformer}!")
        # Get the name of the transformer
        transformer_name = transformer.split("/")[-1]
        embedding_name = f"{transformer_name}_embs.npy"
        embedding_path = create_emb_path(args.embedding_path, embedding_name)
        if embedding_path.exists() and args.lazy:
            print("already processed so we skip!")
            continue
        # Boot it up
        sentence_model = SentenceTransformer(transformer)

        # Embedding the documents #
        print("Embedding documents!")
        all_embeds = sentence_model.encode(all_paragraphs, show_progress_bar=True)

        print("done with embedding...")
        # Writing data to disk
        print(f"written to disk at {embedding_path}!")
        np.save(embedding_path, all_embeds)
        np

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
    my_parser.add_argument(
        "--lazy", action="store_true", help="doesn't transformers if embeddings exist"
    )
    my_parser.add_argument("--chunk-size", type=int, required=False)

    args = my_parser.parse_args()
    main(args)

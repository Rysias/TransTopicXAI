"""
Pipeline for embedding documents. 
This is for making the documents ready for BERTopic 
"""
import numpy as np
import pickle
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List


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


def main(args):
    DATA_DIR = Path("../BscThesisData/data")
    data_path = DATA_DIR / "paragraph_dict.pkl"
    resume_dict = read_pickle(data_path)
    all_paragraphs = flatten_list(list(resume_dict.values()))
    # Powering up the transformer!
    transformer_list = [
        "chcaa/da_dacy_small_trf",
        "chcaa/da_dacy_medium_trf",
        "chcaa/da_dacy_large_trf",
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
        embedding_path = DATA_DIR / f"{transformer_name}_embedding_dict.pkl"
        serialize_data(embedding_dict, embedding_path)
    print("all done!")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Embed Documents")
    args = my_parser.parse_args()
    main(args)

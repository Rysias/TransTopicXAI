import embed_documents_df as edd
import pandas as pd
import random
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_nonsense(word_list, length=5):
    return " ".join(random.choice(word_list) for i in range(length))


model = SentenceTransformer("Maltehb/-l-ctra-danish-electra-small-cased")

word_list = ["hej", "jeg", "er", "en", "ged"]
test_data = [generate_nonsense(word_list) for _ in range(100)]


output = edd.process_roberta_embs(np.array(test_data), model, chunk_size=10)
output.shape

np.all(np.all(output != 0, axis=1))

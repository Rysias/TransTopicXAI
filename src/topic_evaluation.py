""" evaluating coherence for topics """
import pickle
from gensim.models.coherencemodel import CoherenceModel
from pathlib import Path
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def flatten_list(lst):
    return [elem for sublist in lst for elem in sublist]


def get_paragraphs(paragraph_dict):
    return flatten_list(list(paragraph_dict.values()))


def get_model_name(file_path: Path) -> str:
    full_name = file_path.name
    stop_idx = full_name.find("_")
    return full_name[:stop_idx]


DATA_DIR = Path("../../BscThesisData/data")
MODEL_PATH = Path("../models/")


clean_paragraphs = read_pickle(DATA_DIR / "paragraph_dict.pkl")
paragraph_list = get_paragraphs(clean_paragraphs)
vectorizer = read_pickle(MODEL_PATH / "vectorizer.pkl")
X = vectorizer.fit_transform(paragraph_list)

# Tokenizing docs
tokenizer = vectorizer.build_tokenizer()
tokenized_docs = [
    [
        word.lower()
        for word in tokenizer(doc)
        if not word.lower() in vectorizer.stop_words
    ]
    for doc in paragraph_list
]


# Creating dictionary #
# transform sparse matrix into gensim corpus
corpus_vect_gensim = Sparse2Corpus(X, documents_columns=False)
dictionary = Dictionary.from_corpus(
    corpus_vect_gensim,
    id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()),
)


# Evaluating the stuff
topic_paths = list(DATA_DIR.glob("*_topic_dict.pkl"))
for topic_path in topic_paths:
    model_name = get_model_name(topic_path)
    print(f"Evaluating {model_name}")
    topic_dict = read_pickle(topic_path)
    topic_list = list(topic_dict.values())

    cm = CoherenceModel(
        topics=topic_list, coherence="c_v", dictionary=dictionary, texts=tokenized_docs
    )
    print(cm.get_coherence_per_topic())

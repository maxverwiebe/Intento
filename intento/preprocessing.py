import joblib, csv, numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .config import MAX_VOCAB, MAX_LEN, DATA_PATH, TOKEN_PATH, LABEL_PATH, RANDOM_SEED

def load_corpus(csv_path: str | Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["utterance"])
            labels.append(row["intent"])
    return texts, labels


def build_tokenizer(texts: list[str]) -> Tokenizer:
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
    tok.fit_on_texts(texts)
    return tok


def prepare_data() -> tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    texts, labels = load_corpus(DATA_PATH)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    tok = build_tokenizer(texts)
    seqs = tok.texts_to_sequences(texts)
    X    = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")

    # saves the tokenizer and label encoder as joblib files on the disk
    Path(TOKEN_PATH).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(tok, TOKEN_PATH)
    joblib.dump(le, LABEL_PATH)
    return X, y, tok, le
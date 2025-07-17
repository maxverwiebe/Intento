# Intento

A lightweight, production-ready **neural intent-classification framework** for virtual assistants and just experiementing.
Built with TensorFlow/Keras, scikit-learn and pure Python 3.11.

_Imagine asking a voice assistant what time it is?
No matter what combination of Word sentences you use, the assistant usually knows what you mean. This model simulates this._

---

## 1. Overview

Intento turns raw user utterances into imaginary function calls by predicting a single intent label for every input sentence.
Out of the box it ships with:

- a CNN intent classifier (which is fast and CPU-friendly)
- stratified train / validation split
- CLI scripts for training and inference
- reusable tokenizer and label encoder artifacts

## 2. How It Works

### 2.1 Data Pipeline

1. `intento/data/large_test_corpus.csv`
   Two columns:
   - `utterance` free-form user text
   - `intent` target label / fucntion
2. `intento/preprocessing.py`
   - loads CSV
   - fits a `Tokenizer` (num_words = 5000, OOV token "\<UNK>")
   - pads every sequence to `MAX_LEN` tokens (default 20)
   - encodes labels with `LabelEncoder`
   - persists `tokenizer.joblib` and `label_encoder.joblib`

### 2.2 Model Architecture (model.py)

<IMAGE HERE>
```

Input -> Embedding -> 1D-Conv (128 filters, kernel 5, ReLU)
-> GlobalMaxPooling
-> Dense(64, ReLU) -> Dropout(0.3)
-> Dense(num_classes, Softmax) -> Output

- **Embedding layer** transforms token IDs into dense 100-d vectors
- **Conv1D + GlobalMaxPool** acts like an n-gram feature extractor
- **Dense + Dropout** learns non-linear class boundaries
- **Softmax** returns a probability distribution over all intents

### 2.3 Training Loop (train.py)

- stratified 80 / 20 split (train_test_split)
- `sparse_categorical_crossentropy` loss, `accuracy` metric
- `EarlyStopping(patience=5, restore_best_weights=True)`
- artifacts saved to `models/`

### 2.4 Evaluation Metrics

- overall accuracy (default)
- macro F1, balanced accuracy via `utils.py`
- confusion matrix visualisation

## 3. Quick-Start

### 3.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### 3.2 Training

```bash
python -m intento.train
```

Artifacts are stored in `intento/models/`.

### 3.3 Inference

```bash
python -m intento.predict "Play some jazz"
# -> music.play  (0.94)
```

Or inside Python:

```python
from intento.predict import predict_intent
intent, conf = predict_intent("Is it raining tomorrow?")
```

## 4. Configuration (`config.py`)

```
MAX_VOCAB   = 5000
MAX_LEN     = 20
EMBED_DIM   = 100
VAL_SPLIT   = 0.2
RANDOM_SEED = 42
DATA_PATH   = "data/large_test_corpus.csv"
MODEL_PATH  = "models/intent_model.keras"
TOKEN_PATH  = "models/tokenizer.joblib"
LABEL_PATH  = "models/label_encoder.joblib"
```

Override via environment variables or CLI flags if needed.

## 5. Extending the Model

- **BERT / DistilBERT**
  Replace the CNN in `model.py` with a Transformer encoder (`transformers` library) for higher accuracy.
- **Pre-trained word embeddings**
  Load GloVe or FastText vectors into the embedding layer, freeze or fine-tune.
- **Multi-label setup**
  Switch final activation to sigmoid and loss to `binary_crossentropy`.
- **Unknown / fallback intent**
  Add an extra class and choose a confidence threshold, e.g. 0.5.

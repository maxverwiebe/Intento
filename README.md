# Intento

A lightweight, experimental **neural intent-classification framework** for virtual assistants and just testing.
Built with TensorFlow/Keras, scikit-learn and pure Python.

_Imagine asking a voice assistant what time it is?
No matter what combination of words / sentences you use, the assistant usually knows what you mean. This model simulates this._

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

<img width="887" height="381" alt="image" src="https://github.com/user-attachments/assets/173bcb0c-811f-4930-a6ed-b5c57d9fbf0d" />

1: `Input (L0)`
20 integer IDs represent the record fragment.
Example: [23, 57, 248, 4, 71, 0, 0, ...] (zeros are padding to reach 20)
If senetence has more than 20 words, it gets cut off at the end

2: `Embedding (L1)`
Each ID is projected onto a 100-dimensional vector.

3: `1D-Convolution (L2)`
128 filters a kernel size 5 glide over the sequence and recognize local n-gram patterns ("play some jazz", "turn on the", etc.)

4: `Global Max Pooling (L3)`
For each filter, the strongest activation value is picked out -> a 128-d vector, regardless of sentence length.

5: `Dense + ReLU (L4)`
64 artificial neurons mix the 128 features non-linearly.

6: `Dropout 0.3 (L5)`
During training, 30% of the 64 neurons are randomly deactivated, which prevents overfitting

7: `Logits Dense (L6)`
Each of the C output neurons (an intent class) collects weighted signals from the 64 ReLU units.

8: `Softmax (L7)`
Converts raw logits into a probability distribution; highest probability -> predicted function call

In short:
- **Embedding layer** transforms token IDs into dense 100-d vectors
- **Conv1D + GlobalMaxPool** acts like an n-gram feature extractor
- **Dense + Dropout** learns non-linear class boundaries
- **Softmax** returns a probability distribution over all intents

### 2.3 Training Loop (train.py)

- stratified 80 / 20 split (train_test_split)
- `sparse_categorical_crossentropy` loss, `accuracy` metric
- `EarlyStopping(patience=5, restore_best_weights=True)`
- artifacts saved to `models/`

## 3. Quick-Start

### 3.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate OR .venv\Scripts\activate
pip install --upgrade pip
pip install -e .   // in the root of the cloned repo
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

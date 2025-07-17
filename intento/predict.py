import joblib, sys, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .config import MODEL_PATH, TOKEN_PATH, LABEL_PATH, MAX_LEN

model = tf.keras.models.load_model(MODEL_PATH)
Tok   = joblib.load(TOKEN_PATH)
Le    = joblib.load(LABEL_PATH)


def predict_intent(sentence: str):
    seq = Tok.texts_to_sequences([sentence])
    X   = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(X, verbose=0)[0]
    idx   = probs.argmax()
    return Le.inverse_transform([idx])[0], float(probs[idx])

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("👤 Eingabe: ")
    intent, conf = predict_intent(text)
    print(f"→ {intent}  ({conf:.2%})")
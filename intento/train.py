from pathlib import Path
import joblib, numpy as np
from sklearn.model_selection import train_test_split
from .config import VAL_SPLIT, RANDOM_SEED, MODEL_PATH, TOKEN_PATH, LABEL_PATH
from .preprocessing import prepare_data
from .model import build_model


def main():
    X, y, tok, le = prepare_data()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )

    model = build_model(num_classes=len(le.classes_))

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )

    Path(MODEL_PATH).parent.mkdir(exist_ok=True, parents=True)
    model.save(MODEL_PATH)

    print("Done!")

if __name__ == "__main__":
    import tensorflow as tf
    main()
from tensorflow.keras import layers, models
from .config import MAX_VOCAB, MAX_LEN, EMBED_DIM

def build_model(num_classes: int):
    emb_layer = layers.Embedding(input_dim=MAX_VOCAB,
                                 output_dim=EMBED_DIM,
                                 input_length=MAX_LEN)

    model = models.Sequential([
        emb_layer,
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
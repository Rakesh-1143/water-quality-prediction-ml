"""Feed-forward ANN (9-16-8-out) with Xavier or SHAP-guided initialisation (IEEE paper, Fig. 2)."""
import numpy as np
from tensorflow import keras

from config import NN_PARAMS, RANDOM_STATE
from src.shap_init import scale_first_layer


def build_ann(n_features, n_classes=2, seed=RANDOM_STATE, shap_weights=None, params=NN_PARAMS):
    """Binary (sigmoid) when n_classes == 2, else softmax. `shap_weights` -> SHAP-guided init."""
    keras.utils.set_random_seed(seed)
    out = keras.layers.Dense(1, activation="sigmoid") if n_classes == 2 \
        else keras.layers.Dense(n_classes, activation="softmax")
    h1, h2 = params["hidden"]
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(h1, activation="relu", kernel_initializer="glorot_uniform"),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(h2, activation="relu", kernel_initializer="glorot_uniform"),
        out,
    ])
    if shap_weights is not None:
        first = model.layers[0]
        kernel, bias = first.get_weights()
        first.set_weights([scale_first_layer(kernel, shap_weights).astype("float32"), bias])
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]), loss=loss,
                  metrics=["accuracy"])
    return model


def train_ann(model, X_train, y_train, X_val, y_val, params=NN_PARAMS, epochs=None, verbose=0):
    """Train with early stopping on validation loss; returns the Keras History."""
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                         restore_best_weights=True)
    return model.fit(X_train, np.asarray(y_train), validation_data=(X_val, np.asarray(y_val)),
                     epochs=epochs or params["epochs"], batch_size=params["batch_size"],
                     callbacks=[stop], verbose=verbose)


def predict_proba(model, X):
    """Binary: 1-D P(potable). Multiclass: (n, k) probabilities."""
    p = model.predict(X, verbose=0)
    return p.ravel() if p.shape[1] == 1 else p

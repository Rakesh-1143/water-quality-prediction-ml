"""TensorFlow ANN matching Table VIII and Algorithm 3."""
import numpy as np
from config import LEARNING_RATE, BATCH_SIZE, PATIENCE
from wqi.core import input_kernel


def build_ann(weights=None, seed=42, learning_rate=LEARNING_RATE, dropout=.2):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(9,)),
        tf.keras.layers.Dense(16, activation='relu', name='feature_layer',
            kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(8, activation='relu', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    if weights is not None:
        model.get_layer('feature_layer').set_weights([
            input_kernel(weights), np.zeros(16, dtype=np.float32)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_ann(model, X, y, Xval, yval, epochs=100):
    import tensorflow as tf
    # Explicit datasets bound private thread pools on constrained CPU hosts.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    train = tf.data.Dataset.from_tensor_slices((np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32))).shuffle(len(X), seed=42).batch(BATCH_SIZE)
    val = tf.data.Dataset.from_tensor_slices((np.asarray(Xval, dtype=np.float32),
        np.asarray(yval, dtype=np.float32))).batch(BATCH_SIZE)
    history = model.fit(train.with_options(options),
        validation_data=val.with_options(options), epochs=epochs, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                    patience=PATIENCE, restore_best_weights=True)])
    return history.history


def probability(model, X):
    # Direct inference avoids a new tf.data prediction thread pool on every call.
    return np.asarray(model(np.asarray(X, dtype=np.float32), training=False)).reshape(-1)

"""TensorFlow ANN matching Table VIII and Algorithm 3."""
import numpy as np
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
from config import LEARNING_RATE, BATCH_SIZE, PATIENCE
from wqi.core import input_kernel


def build_ann(weights=None, seed=42, learning_rate=LEARNING_RATE, dropout=.2,
              initialization='repeat'):
    import tensorflow as tf
    if initialization not in {'repeat', 'feature_glorot'}:
        raise ValueError('Unknown feature initialization method')
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
        kernel = input_kernel(weights)
        if initialization == 'feature_glorot':
            # Abstract implementation choice, not verified author code.
            # Preserve different signed connections between hidden neurons;
            # normalized feature importance scales their initial variance.
            base = tf.keras.initializers.GlorotUniform(seed=seed)((9, 16)).numpy()
            kernel = base * np.sqrt(9 * np.asarray(weights, dtype=np.float32))[:, None]
        model.get_layer('feature_layer').set_weights([
            kernel, np.zeros(16, dtype=np.float32)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_ann(model, X, y, Xval, yval, epochs=100, seed=42):
    import tensorflow as tf
    # Explicit datasets bound private thread pools on constrained CPU hosts.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    train = tf.data.Dataset.from_tensor_slices((np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32))).shuffle(len(X), seed=seed).batch(BATCH_SIZE)
    val = tf.data.Dataset.from_tensor_slices((np.asarray(Xval, dtype=np.float32),
        np.asarray(yval, dtype=np.float32))).batch(BATCH_SIZE)
    history = model.fit(train.with_options(options),
        validation_data=val.with_options(options), epochs=epochs, verbose=0, shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                    patience=PATIENCE, restore_best_weights=True)])
    return history.history


def probability(model, X):
    # Direct inference avoids a new tf.data prediction thread pool on every call.
    return np.asarray(model(np.asarray(X, dtype=np.float32), training=False)).reshape(-1)

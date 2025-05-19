import tensorflow as tf
import numpy as np
import os

from sklearn.model_selection import train_test_split

INPUT_FOLDER = os.path.abspath('./dataset/input_samples/')
OUTPUT_FOLDER = os.path.abspath('./dataset/samples/')

def build_model(input_size, layers=5):
    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs

    for _ in range(layers - 1):
        x = tf.keras.layers.Dense(units=input_size, activation='linear')(x)

    # Final layer
    x = tf.keras.layers.Dense(units=input_size, activation=None)(x)
    outputs = x
    
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    )

    return model

def normalize(points):
    points = points.reshape(1000, 2)

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    points[:, 0] = (points[:, 0] - min_x) / (max_x - min_x)
    points[:, 1] = (points[:, 1] - min_y) / (max_y - min_y)

    points = points.reshape(2000,)
    return points

def preprocess_fft(x):
    arr = x.reshape(1000, 2)
    arr_complex = arr[:, 0] + 1j * arr[:, 1]
    arr_fft = np.fft.fft(arr_complex)

    mag = np.abs(arr_fft)
    phase = np.angle(arr_fft)

    x = np.concatenate([mag, phase], axis=-1)
    return x

def inverse_preprocess_fft(x):
    """
    Inverts the preprocess_fft function.
    
    Parameters:
        x (np.ndarray): Array of shape (1002,), with first 501 values being magnitudes
                        and last 501 values being phases.
    
    Returns:
        np.ndarray: Reconstructed array of shape (2000,), where each pair of values 
                    represents real and imaginary parts.
    """
    mag = x[:1000]
    phase = x[1000:]

    # Reconstruct the complex spectrum
    signal = mag * np.exp(1j * phase)

    # Inverse FFT to get time-domain complex signal
    arr_complex = np.fft.ifft(signal)

    # Separate real and imaginary parts and flatten
    arr = np.stack([arr_complex.real, arr_complex.imag], axis=-1).reshape(-1)
    return arr

def batchGenerator(files, batch_size=32, preprocess=None):
    indices = np.arange(len(files))
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i: i+batch_size]
            if len(batch_indices) < batch_size:
                continue

            X_batch = []
            Y_batch = []

            for file in batch_indices:
                x = np.load(os.path.join(INPUT_FOLDER, files[file]))
                y = np.load(os.path.join(OUTPUT_FOLDER, files[file]))
                if preprocess:
                    X_batch.append(preprocess(x))
                    Y_batch.append(preprocess(y))

            yield np.array(X_batch), np.array(Y_batch)

if __name__ == '__main__':
    # === CONFIGURACIÓN GENERAL ===
    LOAD_MODEL = True
    MODEL_PATH = "models/NormalContNNv1_500.keras"
    EPOCHS = 1000
    BATCH_SIZE = 64

    if LOAD_MODEL:
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model(input_size=2000, layers=10)

    model.summary()

    # === Batch generation ===
    files = os.listdir(INPUT_FOLDER)
    indices = np.arange(len(files))
    print(len(files))

    id_train, id_temp = train_test_split(indices, test_size=0.3, random_state=42)
    id_val, id_test = train_test_split(id_temp, test_size=0.5, random_state=42)

    files_train = [files[i] for i in id_train]
    files_val = [files[i] for i in id_val]
    files_test = [files[i] for i in id_test]

    print(len(files_train), len(files_val), len(files_test))

    print(files_train)

    # === Entrenamiento ===
    steps_per_epoch = len(files_train) // BATCH_SIZE
    val_steps = len(files_val) // BATCH_SIZE
    print(steps_per_epoch, val_steps)

    train_gen = batchGenerator(files_train, batch_size=BATCH_SIZE, preprocess=normalize)
    val_gen = batchGenerator(files_val, batch_size=BATCH_SIZE, preprocess=normalize)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True, monitor='val_mse', verbose=1
    )

    #early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    #    monitor='val_mse', patience=40, verbose=1
    #)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[checkpoint_cb],# early_stopping_cb],
        verbose=1
    )
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

INPUT_FOLDER = os.path.abspath('./dataset/input_samples/')
OUTPUT_FOLDER = os.path.abspath('./dataset/samples/')

def build_model(input_size, connections=None, layers=5):
    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs

    dense_layers = []
    for i in range(layers - 1):
        layer = tf.keras.layers.Dense(
            units=input_size,
            activation='linear'  # Can change to 'relu' if needed
        )
        x = layer(x)
        dense_layers.append(layer)

    # Final layer
    final_layer = tf.keras.layers.Dense(
        units=input_size,
        activation=None
    )
    x = final_layer(x)
    dense_layers.append(final_layer)

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

    # Manually set weights to identity
    for layer in dense_layers:
        identity = np.eye(input_size, dtype=np.float32)
        bias = np.zeros(input_size, dtype=np.float32)
        layer.set_weights([identity, bias])

    return model

def batchGenerator(files, batch_size=32):
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
                X_batch.append(np.load(os.path.join(INPUT_FOLDER, files[file])))
                Y_batch.append(np.load(os.path.join(OUTPUT_FOLDER, files[file])))

            yield np.array(X_batch), np.array(Y_batch)

def main(load_model, model_path, epochs, batch_size, graph_name):
    if load_model:
        model = tf.keras.models.load_model(model_path)
    else:
        model = build_model(input_size=2000, connections=30, layers=10)

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
    steps_per_epoch = len(files_train) // batch_size
    val_steps = len(files_val) // batch_size
    print(steps_per_epoch, val_steps)

    train_gen = batchGenerator(files_train, batch_size=batch_size)
    val_gen = batchGenerator(files_val, batch_size=batch_size)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='loss', verbose=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse', patience=10, verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )

    # summarize history for accuracy
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Mean Absolute Error')
    plt.ylabel('mae')
    plt.xlabel('época')
    plt.legend(['entrenamiento', 'validación'], loc='upper left')
    plt.savefig(f'plots/{graph_name}_mae.png')

    # clear plot
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Mean Squared Error')
    plt.ylabel('mse')
    plt.xlabel('época')
    plt.legend(['entrenamiento', 'validación'], loc='upper left')
    plt.savefig(f'plots/{graph_name}_mse.png')

if __name__ == '__main__':
    # === CONFIGURACIÓN GENERAL ===
    LOAD_MODEL = False
    MODEL_NAME = 'ContNNv4'
    MODEL_PATH = f"models/{MODEL_NAME}.keras"
    EPOCHS = 2
    BATCH_SIZE = 64

    main(LOAD_MODEL, MODEL_PATH, EPOCHS, BATCH_SIZE, MODEL_NAME)
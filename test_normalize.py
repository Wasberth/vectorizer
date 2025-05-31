import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from FourierContourNN import normalize, preprocess_fft, inverse_preprocess_fft

MODEL = 'models/NormalContNNv2_best.keras'
INPUT_FOLDER = 'dataset/input_samples/'
EXPECTED_FOLDER = 'dataset/samples/'
PREPROCESS = normalize
POSTPROCESS = None
TEST = True
INDEXES = [5, 7, 9]

model = tf.keras.models.load_model(MODEL, compile=False)

# Calculate Metrics
def calculate_metrics(y_true, y_pred):
    mae = tf.keras.metrics.MeanAbsoluteError(name='mae')
    mse = tf.keras.metrics.MeanSquaredError(name='mse')

    mae.update_state(y_true, y_pred)
    mse.update_state(y_true, y_pred)

    return mae.result().numpy(), mse.result().numpy()

def graph(points, expected):
    predicted = model.predict(np.expand_dims(points, axis=0))

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)

    if POSTPROCESS:
        predicted = POSTPROCESS(predicted)
        expected = POSTPROCESS(expected)
        points = POSTPROCESS(points)

    points = points.reshape((-1, 2))
    expected = expected.reshape((-1, 2))
    predicted = predicted.reshape((-1, 2))

    ax1.plot(points[:, 0], points[:, 1], 'o')
    ax1.plot(points[:, 0][0], points[:, 1][0], marker='o', color='r')
    ax1.set_title('Points from input')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True)

    ax2.plot(expected[:, 0], expected[:, 1], 'o')
    ax2.plot(expected[:, 0][0], expected[:, 1][0], marker='o',  color='r')
    ax2.set_title('Points expected')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    ax2.grid(True)

    ax3.plot(predicted[:, 0], predicted[:, 1], 'o')
    ax3.plot(predicted[:, 0][0], predicted[:, 1][0], marker='o',  color='r')
    ax3.set_title('Points predicted')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.axis('equal')
    ax3.grid(True)

    plt.show()

files = os.listdir(INPUT_FOLDER)
indices = np.arange(len(files))
print(len(files))

id_train, id_temp = train_test_split(indices, test_size=0.3, random_state=42)
id_val, id_test = train_test_split(id_temp, test_size=0.5, random_state=42)

files_test = [files[i] for i in id_test]

x = np.zeros((len(files_test), 2000))
y_true = np.zeros((len(files_test), 2000))

for i, file in enumerate(files_test):
    points = np.load(os.path.join(INPUT_FOLDER, file))
    expected = np.load(os.path.join(EXPECTED_FOLDER, file))

    if PREPROCESS:
        points = PREPROCESS(points)
        expected = PREPROCESS(expected)

    x[i, :] = points
    y_true[i, :] = expected

if TEST:
    y_pred = model.predict(x)

    mae, mse = calculate_metrics(y_true, y_pred)
    print('MAE:', mae)
    print('MSE:', mse)

for i in INDEXES:
    graph(x[i], y_true[i])
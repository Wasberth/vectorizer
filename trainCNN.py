import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import os
from tensorflow import optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Flatten, Dense
import tensorflow as tf
import pickle
from utilsCNN import batchGenerator, createFiles
from generateCNNDataset import vector_num

# Todavía puedes:
# Poner 2 modelos, uno para SE y otro para control
tf.keras.backend.set_image_data_format('channels_last')

dataset_directory = 'dataset/'
input_directory = dataset_directory+"inputCNN/"
output_directory = dataset_directory+"outputCNN/"
use_directory = dataset_directory+"filesInUse/"
reference_directory = os.fsencode(dataset_directory+input_directory)
model_directory = 'models/'

def createCleanModel(vector_num):
    input_shape = (500, 500, 1)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        input_w = tf.keras.layers.Input(shape=input_shape)

        w = Conv2D(filters=2, kernel_size=(5,5), kernel_initializer='he_uniform', activation='relu')(input_w)
        w = MaxPooling2D((2,2))(w)
        w = Conv2D(filters=4, kernel_size=(5,5), kernel_initializer='he_uniform', activation='relu')(w)
        w = MaxPooling2D((2,2))(w)
        w = Conv2D(filters=8, kernel_size=(5,5), kernel_initializer='he_uniform', activation='relu')(w)
        w = MaxPooling2D((2,2))(w)
        w = Conv2D(filters=4, kernel_size=(5,5), kernel_initializer='he_uniform', activation='relu')(w)
        w = MaxPooling2D((2,2))(w)
        w = Flatten()(w)
        w = Dense(vector_num*3*64, activation='relu', kernel_initializer='he_uniform')(w)
        w = Dense(vector_num*3*32, activation='relu', kernel_initializer='he_uniform')(w)
        w = Dense(vector_num*3*16, activation='relu', kernel_initializer='he_uniform')(w)
        w = Dense(vector_num*3*8, activation='relu', kernel_initializer='he_uniform')(w)
        w = Dense(vector_num*3*4, activation='relu', kernel_initializer='he_uniform')(w)
        w = Dense(vector_num*3*2, activation='relu', kernel_initializer='he_uniform')(w)

        output = Dense(vector_num*3, activation='tanh')(w)
        model = tf.keras.models.Model(input_w, output)
        loss = tf.keras.losses.MeanSquaredError()
        mae_metric = tf.keras.metrics.MeanAbsoluteError()

        model.compile(optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0), metrics=[mae_metric], loss=loss, run_eagerly=True)
    return model

if __name__ == "__main__":
    # Config
    dataset_sufix = 0
    create_train_val_test_file = True
    load_model = False
    epoch = 1
    batch_size = 256

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[1], True)
        except RuntimeError as e:
            print(e)

    if create_train_val_test_file:
        shapes_dict = createFiles(vector_num)
    
    with open(use_directory+"shapes.pkl", "rb") as dict_file:
        shapes_dict = pickle.load(dict_file)
    
    print(shapes_dict)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    
    X_train = np.memmap(use_directory+'train_input.npy', dtype=np.uint8, mode='r', shape=shapes_dict['train_input'])
    X_val = np.memmap(use_directory+'val_input.npy', dtype=np.uint8, mode='r', shape=shapes_dict['val_input'])
    X_test = np.memmap(use_directory+'test_input.npy', dtype=np.uint8, mode='r', shape=shapes_dict['test_input'])
    Y_train = np.memmap(use_directory+'train_output.npy', mode='r', dtype=np.float64, shape=shapes_dict['train_output'])
    Y_val = np.memmap(use_directory+'val_output.npy', mode='r', dtype=np.float64, shape=shapes_dict['val_output'])
    Y_test = np.memmap(use_directory+'test_output.npy', mode='r', dtype=np.float64, shape=shapes_dict['test_output'])
    
    if load_model:
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            model = tf.keras.models.load_model(model_directory+'CNN.keras')
    else:
        model = createCleanModel(vector_num)
    
    # Modificar el learning rate
    model.optimizer.learning_rate.assign(0.0001)

    metric = 'val_mean_absolute_error'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_directory+'CNN.keras',
                                                          monitor=metric, 
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='max')
    
    train_gen = batchGenerator(X_train, Y_train, batch_size=batch_size)
    val_gen = batchGenerator(X_val, Y_val, batch_size=batch_size)
    
    steps_per_epoch = X_train.shape[0] // batch_size
    val_steps = X_val.shape[0] // batch_size
    
    history = model.fit(train_gen, epochs=epoch, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=val_steps, callbacks=[model_checkpoint], verbose=1)

    with open('models/performance.pkl', 'wb') as performance_file:
        pickle.dump(history.history, performance_file)

    # x = range(1, len(mse) + 1)

    # plt.plot(x, mse, label='Training MSE')
    # plt.plot(x, v_mse, label='Validation MSE')
    # plt.title('Convolutional neural network training loss')
    # plt.legend()
    # plt.figure()
    # plt.plot(x, mae, label='Training MAE')
    # plt.plot(x, v_mae, label='Validation MAE')
    # plt.title('Convolutional neural network training metrics')
    # plt.legend()
    # plt.show()
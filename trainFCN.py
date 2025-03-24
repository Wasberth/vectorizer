import numpy as np
import os
from tensorflow import optimizers
import tensorflow as tf
import pickle
from utilsFCN import batchGenerator, createFiles

fcn_dataset_directory = "D:/Escolar/DatasetFCN/"
input_directory = fcn_dataset_directory+"inputFCN/"
output_directory = fcn_dataset_directory+"outputFCN/"
use_directory = fcn_dataset_directory+"filesInUse/"
reference_directory = os.fsencode(fcn_dataset_directory+input_directory)
dataset_directory = 'C:/Users/sonic/Documents/USB/Escolar/TT/vectorizer/dataset/'
model_directory = 'C:/Users/sonic/Documents/USB/Escolar/TT/vectorizer/models/'
padding = 3

def createCleanModel():
    input_shape = (None, None, 1)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus, tf.distribute.HierarchicalCopyAllReduce())
    with strategy.scope():
        input_w = tf.keras.layers.Input(shape=input_shape)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=5, padding='same', activation='relu')(input_w)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=5, padding='same', activation='relu')(w)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=5, padding='same', activation='relu')(w)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=5, padding='same', activation='relu')(w)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=5, padding='same', activation='relu')(w)
        w = tf.keras.layers.ZeroPadding2D(padding=1)(w)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu')(w)
        w = tf.keras.layers.ZeroPadding2D(padding=1)(w)
        w = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu')(w)
        w = tf.keras.layers.ZeroPadding2D(padding=1)(w)
        output = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='sigmoid')(w)
        model = tf.keras.models.Model(input_w, output)
        mse = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), metrics='accuracy', loss=mse)
    return model

if __name__ == "__main__":
    # Config
    dataset_sufix = 0
    create_train_val_test_file = False
    load_model = False
    epoch = 5
    batch_size = 32

    if create_train_val_test_file:
        createFiles(dataset_sufix)
    
    with open(use_directory+"shapes.pkl", "rb") as dict_file:
        shapes_dict = pickle.load(dict_file)
    
    X_train = np.memmap(use_directory+'train_input.npy', mode='r', shape=shapes_dict['train_input'])
    X_val = np.memmap(use_directory+'val_input.npy', mode='r', shape=shapes_dict['val_input'])
    X_test = np.memmap(use_directory+'test_input.npy', mode='r', shape=shapes_dict['test_input'])
    Y_train = np.memmap(use_directory+'train_output.npy', mode='r', shape=shapes_dict['train_output'])
    Y_val = np.memmap(use_directory+'val_output.npy', mode='r', shape=shapes_dict['val_output'])
    Y_test = np.memmap(use_directory+'test_output.npy', mode='r', shape=shapes_dict['test_output'])

    print(X_train.shape, Y_train.shape)
    print(X_val.shape, Y_val.shape)
    print(X_test.shape, Y_test.shape)

    if load_model:
        model = tf.keras.models.load_model(model_directory+'FCN.keras')
    else:
        model = createCleanModel()
        
    metric = 'val_accuracy'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_directory+'FCN.keras',
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
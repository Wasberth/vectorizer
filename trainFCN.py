import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import os
from tensorflow import optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Conv2DTranspose, Concatenate
import tensorflow as tf
import pickle
from utilsFCN import batchGenerator, createFiles

# Todavía puedes:
# Formatear el dataset channels_first
# Hacer líneas en vez de puntos
# Cambiar la función de pérdida
# Cambiar la capa final por tanh (y poner -1's en vez de 0's)
tf.keras.backend.set_image_data_format('channels_last')

dataset_directory = 'dataset/'
input_directory = dataset_directory+"inputFCN/"
output_directory = dataset_directory+"outputFCN/"
use_directory = dataset_directory+"filesInUse/"
reference_directory = os.fsencode(dataset_directory+input_directory)
model_directory = '/home/r1_tocayo/Documents/Escolar/TT/vectorizer/models/'
padding = 3

def conv2d_bn(input_tensor, n_filters, kernel_size):
    data_format = 'channels_last'
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal')(x)
    return x

def createCleanModel():
    input_shape = (1440, 1088, 1)
    n_filters = 32
    dropout = 0.05
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        input_w = tf.keras.layers.Input(shape=input_shape)

        # Contracting path
        c1 = conv2d_bn(input_w, n_filters*1, 3)
        p1 = MaxPooling2D((2,2))(c1)
        p1 = Dropout(dropout*0.5)(p1)
        c2 = conv2d_bn(p1, n_filters*2, 3)
        p2 = MaxPooling2D((2,2))(c2)
        p2 = Dropout(dropout*0.5)(p2)
        c3 = conv2d_bn(p2, n_filters*4, 3)
        p3 = MaxPooling2D((2,2))(c3)
        p3 = Dropout(dropout*0.5)(p3)
        c4 = conv2d_bn(p3, n_filters*8, 3)
        p4 = MaxPooling2D((2,2))(c4)
        p4 = Dropout(dropout*0.5)(p4)
        c5 = conv2d_bn(p4, n_filters*16, 3)

        # Expansive path
        u6 = Conv2DTranspose(filters=n_filters*8, kernel_size=(3,3), strides=(2,2), padding='same')(c5)
        u6 = Concatenate(axis=-1)([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_bn(u6, n_filters*8, kernel_size=3)
        u7 = Conv2DTranspose(filters=n_filters*4, kernel_size=(3,3), strides=(2,2), padding='same')(c6)
        u7 = Concatenate(axis=-1)([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_bn(u7, n_filters*4, kernel_size=3)
        u8 = Conv2DTranspose(filters=n_filters*2, kernel_size=(3,3), strides=(2,2), padding='same')(c7)
        u8 = Concatenate(axis=-1)([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_bn(u8, n_filters*2, kernel_size=3)
        u9 = Conv2DTranspose(filters=n_filters*1, kernel_size=(3,3), strides=(2,2), padding='same')(c8)
        u9 = Concatenate(axis=-1)([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_bn(u9, n_filters*1, kernel_size=3)

        output = Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(c9)
        model = tf.keras.models.Model(input_w, output)
        loss = tf.keras.losses.BinaryFocalCrossentropy()
        accuracy = tf.keras.metrics.Accuracy()
        recall = tf.keras.metrics.Recall()
        iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])

        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), metrics=[accuracy, recall, iou], loss=loss)
    return model

if __name__ == "__main__":
    # Config
    dataset_sufix = 0
    create_train_val_test_file = False
    load_model = False
    epoch = 1
    batch_size = 2

    if create_train_val_test_file:
        shapes_dict = createFiles(dataset_sufix)
    
    with open(use_directory+"shapes.pkl", "rb") as dict_file:
        shapes_dict = pickle.load(dict_file)
    
    print(shapes_dict)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    
    X_train = np.memmap(use_directory+'train_input.npy', mode='r', shape=shapes_dict['train_input'])
    X_val = np.memmap(use_directory+'val_input.npy', mode='r', shape=shapes_dict['val_input'])
    X_test = np.memmap(use_directory+'test_input.npy', mode='r', shape=shapes_dict['test_input'])
    Y_train = np.memmap(use_directory+'train_output.npy', mode='r', shape=shapes_dict['train_output'])
    Y_val = np.memmap(use_directory+'val_output.npy', mode='r', shape=shapes_dict['val_output'])
    Y_test = np.memmap(use_directory+'test_output.npy', mode='r', shape=shapes_dict['test_output'])
    
    # print(X_train.shape, Y_train.shape)
    # print(X_val.shape, Y_val.shape)
    # print(X_test.shape, Y_test.shape)
    
    if load_model:
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
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

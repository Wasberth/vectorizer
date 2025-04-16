import numpy as np
import os
import tensorflow as tf
import pickle
from utilsFCN import batchGenerator

fcn_dataset_directory = "dataset/"
model_directory = 'models/'
use_directory = fcn_dataset_directory+"filesInUse/"

if __name__ == "__main__":
    batch_size = 32

    with open(use_directory+'shapes.pkl', 'rb') as dict_file:
        shape_dict = pickle.load(dict_file)
    test_input_shape = shape_dict['test_input']
    test_output_shape = shape_dict['test_output']
    model = tf.keras.models.load_model(model_directory+"FCN.keras")
    test_input = np.memmap(use_directory+"test_input.npy", mode='r', shape=test_input_shape)
    test_output = np.memmap(use_directory+'test_output.npy', mode='r', shape=test_output_shape)

    steps = np.ceil(test_input_shape[0]/batch_size)

    test_gen = batchGenerator(test_input, test_output, batch_size=batch_size)

    history = model.evaluate(test_gen, steps=steps, verbose=1)
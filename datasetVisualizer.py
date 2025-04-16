import numpy as np
import matplotlib.pyplot as plt
import utilsFCN
import tensorflow as tf
import pickle

from generateFCNDataset import concatMatrixFCN, root, parseColor, color_keyword_dict
from svgpathtools import svg2paths
import os
from PIL import Image
import re

def show_predictions(expected, predicted, channel):
    fig, (ax1, ax2) = plt.subplots(1,2)
    colors = [(0,0,0), (255,255,255)]
    ax1.set_title('Canal esperado')
    ax2.set_title('Canal predicho')
    ax1.imshow(utilsFCN.drawImageFromArray(expected[channel], colors))
    ax2.imshow(utilsFCN.drawImageFromArray(predicted[channel], colors))
    plt.show()

def visualize_model(predict_id):
    with open(utilsFCN.use_directory+'shapes.pkl', 'rb') as dict_file:
        shape_dict = pickle.load(dict_file)
    test_input_shape = shape_dict['test_input']
    test_output_shape = shape_dict['test_output']
    model = tf.keras.models.load_model(utilsFCN.model_directory+'FCN.keras')

    test_input = np.memmap(utilsFCN.use_directory+'test_input.npy', mode='r', shape=test_input_shape)
    test_output = np.memmap(utilsFCN.use_directory+'test_output.npy', mode='r', shape=test_output_shape)

    test_img = test_input[predict_id]
    test_img = np.expand_dims(test_img, axis=0)

    predicted_output = model.predict(test_img)
    predicted_output = np.transpose(predicted_output[0], (2, 0, 1))
    test = np.transpose(test_output[predict_id], (2,0,1))

    show_predictions(test, predicted_output, 0)
    show_predictions(test, predicted_output, 1)
    show_predictions(test, predicted_output, 2)

def visualize_img(test_id):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    dataset_sufix = 0
    file = utilsFCN.input_directory+f'{dataset_sufix}.npy'
    input_shape = utilsFCN.getInputShape(file)
    input = np.memmap(file, mode='r', shape=input_shape)
    output_shape = (input_shape[0], input_shape[1], input_shape[2], 3)
    file = utilsFCN.output_directory+f'{dataset_sufix}.npy'
    output = np.memmap(file, mode='r', shape=output_shape)

    test = np.transpose(input[test_id], (2, 0, 1))
    colors = [(0,0,0), (255, 255, 255)]

    ax1.set_title("Imagen original")
    ax2.set_title("Pixeles SE")
    ax3.set_title("Pixeles Control")
    ax4.set_title("Pixeles basura")
    ax1.imshow(utilsFCN.drawImageFromArray(test[0], colors))
    test = np.transpose(output[test_id], (2, 0, 1))
    colors = [(0,0,0), (255, 255, 255)]
    ax2.imshow(utilsFCN.drawImageFromArray(test[0], colors))
    ax3.imshow(utilsFCN.drawImageFromArray(test[1], colors))
    ax4.imshow(utilsFCN.drawImageFromArray(test[2], colors))
    plt.show()

def validate_again(output_matrix, paths, attributes, color_classes):
    vector_num = 0
    control_num = 0
    for i in range(len(paths)):
        classes = attributes[i]["class"].split()
        is_none = True
        usefull_class = classes[0]
        for c in classes:
            if(c in color_classes.keys()):
                is_none = False
                usefull_class = c
        if(is_none):
            continue
        for curve in paths[i]:
            if(usefull_class != 'fil0'):
                vector_num += 1

                if(hasattr(curve, 'control1')):
                    control_num += 1
                if(hasattr(curve, 'control2')):
                    control_num += 1

    valid = False
    se_img_num = 0
    control_img_num = 0
    for i in range(output_matrix.shape[0]):
        for j in range(output_matrix.shape[1]):
            if output_matrix[i][j][0] == 1:
                se_img_num += 1
                if not valid:
                    print('Al menos tiene un pixel en el canal SE')
                valid = True
            if output_matrix[i][j][1] == 1:
                control_img_num += 1

    if se_img_num != vector_num:
        print('No tiene la misma cantidad de pixeles como de vectores')
    
    if control_img_num != control_num:
        print('No tiene la misma cantidad de pixeles control que cantidad de puntos control')

def visualize_generation(file_id, dataset_sufix):
    directory = os.fsencode(root+'output/')
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        if filename.endswith(f'_{dataset_sufix}'):
            if i != file_id:
                i += 1
                continue
            input_file = root+'input/'+filename+'.png'
            output_file = root+'output/'+os.fsdecode(file)
            break
    
    img = Image.open(input_file)
    paths, attributes = svg2paths(output_file)
    color_classes = {}

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            x = re.findall(r"\.fil[0-9]+", line)
            if x:
                key = x[0][1:]
                y = re.findall(r"\#[0-9A-F]{6}", line)
                if y:
                    color_classes[key] = parseColor(y[0])
                else:
                    y = re.search(r"{fill:(\w+)", line)
                    if(y and y.group(1) != "none"):
                        color_classes[key] = parseColor(color_keyword_dict[y.group(1)])
    
    input_matrix, output_matrix, valid = concatMatrixFCN(img, paths, attributes, color_classes)

    if valid:
        print('Esta es una imagen valida')
    else:
        print('Hay un problema con la imagen')
        validate_again(output_matrix, paths, attributes, color_classes)
    
    input_matrix = np.transpose(input_matrix, (2,0,1))
    output_matrix = np.transpose(output_matrix, (2,0,1))
    colors = [(0,0,0), (255,255,255)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.set_title("Imagen original")
    ax2.set_title("Pixeles SE")
    ax3.set_title("Pixeles Control")
    ax4.set_title("Pixeles basura")
    ax1.imshow(utilsFCN.drawImageFromArray(input_matrix[0], colors))
    ax2.imshow(utilsFCN.drawImageFromArray(output_matrix[0], colors))
    ax3.imshow(utilsFCN.drawImageFromArray(output_matrix[1], colors))
    ax4.imshow(utilsFCN.drawImageFromArray(output_matrix[2], colors))
    plt.show()

# visualize_img(10)
# visualize_model(0)
visualize_generation(0, 0)
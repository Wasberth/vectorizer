import numpy as np
import matplotlib.pyplot as plt
import utilsFCN
import utilsCNN
import tensorflow as tf
import pickle

from generateFCNDataset import concatMatrixFCN, root, parseColor, color_keyword_dict
from svgpathtools import svg2paths
import os
from PIL import Image
import re

from generateCNNDataset import concatMatrixCNN

def show_predictions(expected, predicted, channel):
    fig, (ax1, ax2) = plt.subplots(1,2)
    colors = [(0,0,0), (255,255,255)]
    ax1.set_title('Canal esperado')
    ax2.set_title('Canal predicho')
    ax1.imshow(utilsFCN.drawImageFromArray(expected[channel], colors))
    ax2.imshow(utilsFCN.drawImageFromArray(predicted[channel], colors))
    plt.show()

# def visualize_model(predict_id):
#     with open(utilsFCN.use_directory+'shapes.pkl', 'rb') as dict_file:
#         shape_dict = pickle.load(dict_file)
#     test_input_shape = shape_dict['test_input']
#     test_output_shape = shape_dict['test_output']
#     model = tf.keras.models.load_model(utilsFCN.model_directory+'FCN.keras')
# 
#     test_input = np.memmap(utilsFCN.use_directory+'test_input.npy', mode='r', shape=test_input_shape)
#     test_output = np.memmap(utilsFCN.use_directory+'test_output.npy', mode='r', shape=test_output_shape)
# 
#     test_img = test_input[predict_id]
#     test_img = np.expand_dims(test_img, axis=0)
# 
#     predicted_output = model.predict(test_img)
#     predicted_output = np.transpose(predicted_output[0], (2, 0, 1))
#     test = np.transpose(test_output[predict_id], (2,0,1))
# 
#     show_predictions(test, predicted_output, 0)
#     show_predictions(test, predicted_output, 1)
#     show_predictions(test, predicted_output, 2)

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
    input_file = f'{root}input/{file_id}_{dataset_sufix}.png'
    output_file = f'{root}output/{file_id}_{dataset_sufix}.svg'
    
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

def visualize_windowed_image(file_id, dataset_sufix, window, vector_num):
    # Ya me dio flojera sacar el shape
    input_shape = (106008, window[0], window[1], 1)
    output_shape = (106008, vector_num*3)
    input_matrix = np.memmap(root+'inputCNN/'+str(dataset_sufix)+'.npy', dtype=np.uint8, mode='r', shape=input_shape)
    output_matrix = np.memmap(root+'outputCNN/'+str(dataset_sufix)+'.npy', dtype=np.float64, mode='r', shape=output_shape)
    test_img = input_matrix[file_id]
    test_output = output_matrix[file_id]
    colors = [(0,0,0), (255,255,255)]
    x_test = []
    y_test = []
    for i in range(test_output.shape[0] // 3):
        if test_output[i*3] == 0:
            continue
        x_test.append(test_output[i*3 + 1])
        y_test.append(-test_output[i*3 + 2])

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Entrada")
    ax2.set_title("Salida")
    ax1.imshow(utilsFCN.drawImageFromArray(test_img, colors))
    ax2.plot(x_test, y_test, 'o')
    plt.show()

def visualize_windowed_generation(test_input, test_output, window):
    x_test = []
    y_test = []
    for i in range(test_output.shape[0]):
        x_test.append([])
        y_test.append([])
        padding_y = (i % 4) * window[0]
        padding_x = (i // 4) * window[1]
        for j in range(test_output.shape[1] // 3):
            if test_output[i][j*3] == 0:
                continue
            x_test[i].append(test_output[i][j*3+1] + padding_x)
            y_test[i].append(-test_output[i][j*3+2] - padding_y)

    test_img = np.zeros((window[0]*4, window[1]*3))
    for i in range(3):
        for j in range(4):
            for k in range(window[0]):
                for l in range(window[1]):
                    test_img[window[0]*j + k][window[1]*i + l] = test_input[(j + i*4)][k][l] * (j + i*4)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Entrada")
    ax2.set_title("Salida")
    colors = [(0,0,0),
              (255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (255, 0, 255),
              (0, 255, 255),
              (127, 0, 0),
              (0, 127, 0),
              (0, 0, 127),
              (127, 255, 0),
              (127, 0, 255),
              (255,255,255)]
    ax1.imshow(utilsFCN.drawImageFromArray(test_img, colors))
    for i in range(len(x_test)):
        ax2.plot(x_test[i], y_test[i], 'o')
    plt.show()

    show_id = 6
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Entrada")
    ax2.set_title("Salida")
    ax1.imshow(utilsFCN.drawImageFromArray(test_input[show_id], [(0,0,0), (255,255,255)]))
    ax2.plot(x_test[show_id], y_test[show_id], 'o')
    plt.show()

def rebuild_image(file_id, dataset_sufix, window, vector_num):
    input_shape = (106008, window[0], window[1], 1)
    output_shape = (106008, vector_num*3)
    input_matrix = np.memmap(root+'inputCNN/'+str(dataset_sufix)+'.npy', dtype=np.uint8, mode='r', shape=input_shape)
    output_matrix = np.memmap(root+'outputCNN/'+str(dataset_sufix)+'.npy', dtype=np.float64, mode='r', shape=output_shape)
    test_img = input_matrix[file_id:file_id+12]
    test_output = output_matrix[file_id:file_id+12]
    complete_image = np.zeros((window[0]*4, window[1]*3))
    complete_coords_x = []
    complete_coords_y = []
    for i in range(4):
        for j in range(3):
            for k in range(window[0]):
                for l in range(window[1]):
                    complete_image[window[0]*i + k][window[1]*j + l] = test_img[(i + j*4)][k][l]
    for i in range(test_output.shape[0]):
        x_padding = (i % 4) * window[0]
        y_padding = (i // 4) * window[1]
        for j in range(test_output.shape[1] // 3):
            if test_output[i][j*3] == 0:
                continue
            x = x_padding + test_output[i][j*3 + 1]
            y = -y_padding - test_output[i][j*3 + 2]
            complete_coords_x.append(x)
            complete_coords_y.append(y)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Entrada")
    ax2.set_title("Salida")
    ax1.imshow(utilsFCN.drawImageFromArray(complete_image, [(0,0,0), (255,255,255)]))
    ax2.plot(complete_coords_x, complete_coords_y, 'o')
    plt.show()

def visualize_img_CNN(file_id, width, height, vector_num):
    # Ya me dio flojera sacar el shape
    file_num = 47148
    input_shape = (file_num, width, height, 1)
    output_shape = (file_num, vector_num*3)
    input_matrix = np.memmap(root+'inputCNN/0.npy', dtype=np.uint8, mode='r', shape=input_shape)
    output_matrix = np.memmap(root+'outputCNN/0.npy', dtype=np.float64, mode='r', shape=output_shape)
    shown = False
    showable = True
    test_img = None
    test_output = None
    while not shown:
        if file_id >= file_num:
            file_id = 0
            if showable:
                showable = False
            else:
                print('No se pudo mostrar ninguna imagen')
                return
        print(f'Intentando con {file_id}')
        test_img = input_matrix[file_id]
        test_output = output_matrix[file_id]
        file_id += 1
        colors = [(0,0,0), (255,255,255)]
        x_test = []
        y_test = []
        for i in range(test_output.shape[0] // 3):
            if test_output[i*3] == 0:
                continue
            shown = True
            x_test.append(test_output[i*3 + 1])
            y_test.append(-test_output[i*3 + 2])

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Entrada")
    ax2.set_title("Salida")
    ax1.imshow(utilsFCN.drawImageFromArray(test_img, colors))
    ax2.plot(x_test, y_test, 'o')
    plt.show()

def visualize_CNN_prediction(predict_id, dataset):
    with open(utilsFCN.use_directory+'shapes.pkl', 'rb') as dict_file:
        shape_dict = pickle.load(dict_file)
    input_shape = shape_dict[f'{dataset}_input']
    output_shape = shape_dict[f'{dataset}_output']
    input_matrix = np.memmap(utilsCNN.use_directory+f'{dataset}_input.npy', dtype=np.uint8, mode='r', shape=input_shape)
    output_matrix = np.memmap(utilsCNN.use_directory+f'{dataset}_output.npy', dtype=np.float64, mode='r', shape=output_shape)
    test_input = input_matrix[predict_id]
    test_input = np.expand_dims(test_input, axis=0)
    test_output = output_matrix[predict_id]
    model = tf.keras.models.load_model(utilsCNN.model_directory+'CNN.keras')
    predicted_output = model.predict(test_input)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.imshow(utilsCNN.drawImageFromArray(test_input[0], [(0,0,0), (255, 255, 255)]))
    ax1.set_title("Entrada")
    valid = False
    valid_output = False
    x = []
    y = []
    for i in range(test_output.shape[0]):
        if i % 3 == 0:
            if test_output[i] == -1:
                valid = False
            else:
                valid_output = True
                valid = True
        if i % 3 == 1:
            if valid:
                x.append(test_output[i])
        if i % 3 == 2:
            if valid:
                y.append(test_output[i])
    ax2.plot(x, y, 'o')
    ax2.set_title("Salida esperada")
    valid = False
    x = []
    y = []
    for i in range(predicted_output.shape[1]):
        if i % 3 == 0:
            if predicted_output[0][i] == -1:
                valid = False
            else:
                valid = True
        if i % 3 == 1:
            if valid:
                x.append(predicted_output[0][i])
        if i % 3 == 2:
            if valid:
                y.append(predicted_output[0][i])
    print(predicted_output)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    ax3.plot(x, y, 'o')
    ax3.set_title("Salida predicha")
    plt.show()
    if not valid_output:
        print("No hay una coordenada util en la salida")

# visualize_img(10)
# visualize_model(0)
# visualize_generation('02 con moto_fig', 0)
# visualize_windowed_image(12, 0, (400,400), 66724)

# def visualize_CNN_generation(input_matrix, output_matrix):
#     x = []
#     y = []
#     for i in range(output_matrix.shape[0] // 3):
#         if output_matrix[i*3] == -1:
#             continue
#         x.append(output_matrix[(i*3) + 1])
#         y.append(-output_matrix[(i*3) + 2])
#     fig, (ax1, ax2) = plt.subplots(1,2)
#     ax1.set_title("Entrada")
#     ax1.imshow(utilsCNN.drawImageFromArray(input_matrix, [(0,0,0), (255, 255, 255)]))
#     ax2.set_title("Salida")
#     ax2.set_xlim(0,1)
#     ax2.set_ylim(-1,0)
#     ax2.plot(x, y, 'o')
#     plt.show()
# 
# file = '02 con moto_fig_0'
# 
# img = Image.open(f'dataset/input/{file}.png')
# paths, attributes = svg2paths(f'dataset/output/{file}.svg')
# color_classes = {}
# with open(f'dataset/output/{file}.svg', 'r', encoding='utf-8') as f:
#     for line in f:
#         x = re.findall(r"\.fil[0-9]+", line)
#         if x:
#             key = x[0][1:]
#             y = re.findall(r"\#[0-9A-F]{6}", line)
#             if y:
#                 color_classes[key] = parseColor(y[0])
#             else:
#                 y = re.search(r"{fill:(\w+)", line)
#                 if(y and y.group(1) != "none"):
#                     color_classes[key] = parseColor(color_keyword_dict[y.group(1)])
# 
# input_matrix, output_matrix, valid = concatMatrixCNN(img, paths, attributes, color_classes, 0)
# 
# print(np.min(output_matrix), np.max(output_matrix))
# ind = np.argmax(output_matrix)
# max_matrix = np.unravel_index(ind, output_matrix.shape)
# print(max_matrix)
# visualize_CNN_generation(input_matrix, output_matrix)


# rebuild_image(0, 0, (400, 400), 66724)

# visualize_img_CNN(38677, 500, 500, 1146)

# visualize_CNN_prediction(100, 'test')


def visualize_img_Transformer(file_id, width, height, vector_num):
    # Ya me dio flojera sacar el shape
    file_num = 36427
    input_shape = (file_num, width, height, 1)
    output_shape = (file_num, vector_num*3, 2)
    input_matrix = np.memmap(root+'inputTransformer/0.npy', dtype=np.uint8, mode='r', shape=input_shape)
    output_matrix = np.memmap(root+'outputTransformer/0.npy', dtype=np.float64, mode='r', shape=output_shape)
    shown = False
    showable = True
    test_img = None
    test_output = None
    while not shown:
        if file_id >= file_num:
            file_id = 0
            if showable:
                showable = False
            else:
                print('No se pudo mostrar ninguna imagen')
                return
        print(f'Intentando con {file_id}')
        test_img = input_matrix[file_id]
        test_output = output_matrix[file_id]
        file_id += 1
        colors = [(0,0,0), (255,255,255)]
        x_test = []
        y_test = []
        for i in range(test_output.shape[0]):
            if test_output[i][0] == -1 and test_output[i][1] == -1:
                continue
            shown = True
            x_test.append(test_output[i][0])
            y_test.append(-test_output[i][1])

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Entrada")
    ax2.set_title("Salida")
    ax1.imshow(utilsFCN.drawImageFromArray(test_img, colors))
    ax2.plot(x_test, y_test, 'o')
    plt.show()

visualize_img_Transformer(0, 500, 500, 14)
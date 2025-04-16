import numpy as np
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

dataset_directory = 'dataset/'
input_directory = dataset_directory+"inputFCN/"
output_directory = dataset_directory+"outputFCN/"
use_directory = dataset_directory+"filesInUse/"
model_directory = 'models/'
mod = 16

def drawImageFromArray(array, colors):
    img = Image.new("RGB", (array.shape[1], array.shape[0]), colors[-1])
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if(array[y][x] !=  0):
                i = int(array[y][x])-1
                img.putpixel((x, y), colors[i])
    return img

def parseColor(color_str):
    if color_str.startswith("#"):
        color_str = color_str.lstrip("#")
        if len(color_str) == 3:
            color_str = "".join(c*2 for c in color_str)
        return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
    elif color_str.startswith("rgb"):
        return tuple(map(int, re.findall(r"\d+", color_str)))
    return (-1, -1, -1)

def compareColors(color1, color2):
    if(color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2]):
        return True
    return False

def findColors(file):
    color_amount = 0

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            x = re.findall(r"\.fil[0-9]+", line)
            if(x):
                key = x[0][1:]
                y = re.findall(r"\#[0-9A-F]{6}", line)
                if(y):
                    color_amount += 1
                else:
                    y = re.search(r"{fill:(\w+)", line)
                    if(y and y.group(1) != "none"):
                        color_amount += 1
    return color_amount

def getInputShape(file):
    width = 0
    height = 0
    channels = 1
    num_matrix = 0
    directory = os.fsencode(dataset_directory+'output/')
    file_sufix = file[-5:-4]
    for f in os.listdir(directory):
        file_name = os.fsdecode(f)[:-4]
        if file_name.endswith('_'+file_sufix):
            num_matrix += 1
            if width == 0 or height == 0:
                img = Image.open(dataset_directory+"input/"+file_name+".png")
                width, height = img.size
    height = height + mod + (height % mod)
    width = width + mod + (width % mod)
    return (num_matrix, height, width, channels)

def saveLowRAM(directory, file, ids, original):
    shape = (len(ids), original.shape[1], original.shape[2], original.shape[3])
    save_matrix = np.memmap(directory+file+'.npy', mode='w+', shape=shape)
    i = 0
    for id in ids:
        save_matrix[i] = original[id]
        save_matrix.flush()
        i += 1
    return shape

def createFiles(dataset_sufix):
    input_shape = getInputShape(input_directory+f"{dataset_sufix}.npy")
    X = np.memmap(input_directory+f"{dataset_sufix}.npy", mode="r", shape=input_shape)
    output_shape = (input_shape[0], input_shape[1], input_shape[2], 3)
    Y = np.memmap(output_directory+f"{dataset_sufix}.npy", mode="r", shape=output_shape)

    shapes_dict = {}

    indices = np.arange(X.shape[0])
    id_train, id_temp = train_test_split(indices, test_size=0.3, random_state=42)
    id_val, id_test = train_test_split(id_temp, train_size=0.5, random_state=42)

    shapes_dict["train_input"] = saveLowRAM(use_directory, 'train_input', id_train, X)
    shapes_dict["val_input"] = saveLowRAM(use_directory, 'val_input', id_val, X)
    shapes_dict["test_input"] = saveLowRAM(use_directory, 'test_input', id_test, X)
    shapes_dict["train_output"] = saveLowRAM(use_directory, 'train_output', id_train, Y)
    shapes_dict["val_output"] = saveLowRAM(use_directory, 'val_output', id_val, Y)
    shapes_dict["test_output"] = saveLowRAM(use_directory, 'test_output', id_test, Y)
    
    with open(use_directory+"shapes.pkl", "wb") as dict_file:
        pickle.dump(shapes_dict, dict_file)
    
    return shapes_dict

def batchGenerator(X, Y, batch_size=32):
    indices = np.arange(X.shape[0])
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i: i+batch_size]
            if len(batch_indices) < batch_size:
                continue
            X_batch = np.copy(X[batch_indices])
            Y_batch = np.copy(Y[batch_indices])
            yield X_batch, Y_batch

def checkLimits(coordenate, limits):
    if(coordenate < limits[0]):
        limits[0] = coordenate
    if(coordenate > limits[1]):
        limits[1] = coordenate
    return limits

def imagToPixel(point, height, width, xlim, ylim):
    factor = width/(xlim[1]-xlim[0])
    x = int((point.real-xlim[0])*factor)
    factor = height/(ylim[1]-ylim[0])
    y = int((point.imag-ylim[0])*factor)
    return [x, y]

def checkPadding(coordenate, padding, limit):
    if(coordenate < 0):
        if(0-coordenate > padding):
            return 0-coordenate
    if(coordenate-limit > padding):
        return coordenate-limit
    return padding
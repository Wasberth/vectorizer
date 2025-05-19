import numpy as np
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

dataset_directory = 'dataset/'
input_directory = dataset_directory+"inputCNN/"
output_directory = dataset_directory+"outputCNN/"
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

def getInputShape():
    with open(dataset_directory+'base_file_shapes.pkl', mode='rb') as base_shape_file:
        shape_dict = pickle.load(base_shape_file)
    return shape_dict['input']

def saveLowRAM(directory, file, ids, original, io_type):
    if io_type == 0:
        shape = (len(ids), original.shape[1], original.shape[2], original.shape[3])
        save_matrix = np.memmap(directory+file+'.npy', dtype=np.uint8, mode='w+', shape=shape)
    elif io_type == 1:
        shape = (len(ids), original.shape[1])
        save_matrix = np.memmap(directory+file+'.npy', dtype=np.float64, mode='w+', shape=shape)
    i = 0
    for id in ids:
        save_matrix[i] = original[id]
        save_matrix.flush()
        i += 1
    return shape

def createFiles(vector_num):
    input_shape = getInputShape()
    X = np.memmap(input_directory+f"0.npy", dtype=np.uint8, mode="r", shape=input_shape)
    output_shape = (input_shape[0], vector_num*3)
    Y = np.memmap(output_directory+f"0.npy", dtype=np.float64, mode="r", shape=output_shape)

    shapes_dict = {}

    indices = np.arange(X.shape[0])
    id_train, id_temp = train_test_split(indices, test_size=0.3, random_state=42)
    id_val, id_test = train_test_split(id_temp, train_size=0.5, random_state=42)

    shapes_dict["train_input"] = saveLowRAM(use_directory, 'train_input', id_train, X, 0)
    shapes_dict["val_input"] = saveLowRAM(use_directory, 'val_input', id_val, X, 0)
    shapes_dict["test_input"] = saveLowRAM(use_directory, 'test_input', id_test, X, 0)
    shapes_dict["train_output"] = saveLowRAM(use_directory, 'train_output', id_train, Y, 1)
    shapes_dict["val_output"] = saveLowRAM(use_directory, 'val_output', id_val, Y, 1)
    shapes_dict["test_output"] = saveLowRAM(use_directory, 'test_output', id_test, Y, 1)
    
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

def get_valid_indices(memmap_file, shape, axis, dtype):
    matrix = np.memmap(memmap_file, dtype=dtype, mode='r', shape=shape)
    valid_mask = ~np.all(matrix == 0, axis=axis)
    valid_indices = np.where(valid_mask)[0]
    return valid_indices

def create_from_indices(memmap_file, shape, dtype, valid_indices):
    matrix = np.memmap(memmap_file, dtype=dtype, mode='r', shape=shape)
    filtered_shape = []
    filtered_shape.append(len(valid_indices))
    for i in range(len(matrix.shape)-1):
        filtered_shape.append(matrix.shape[i+1])
    memmap_file = memmap_file[:-4]
    filtered_matrix = np.memmap(memmap_file+'_filtered.npy', dtype=dtype, mode='w+', shape=filtered_shape)

    for i, idx in enumerate(valid_indices):
        filtered_matrix[i] = matrix[idx]
    filtered_matrix.flush()

def sample_curve(curve, n_points=100):
    P0 = curve[0]
    P1 = curve[1]
    P2 = curve[2]
    P3 = curve[3]
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    return ((1 - t)**3) * P0 + 3*((1 - t)**2)*t * P1 + 3*(1 - t)*t**2 * P2 + t**3 * P3

def controls_on_line(line):
    c1 = line[0] + (line[1] - line[0]) / 3
    c2 = line[0] + 2 * (line[1] - line[0]) / 3
    return c1, c2

def controls_on_squared(p0, p1, p2):
    c1 = (1/3) * p0 + (2/3) * p1
    c2 = (2/3) * p1 + (1/3) * p2
    return c1, c2

def discrete_frechet(P, Q):
    n, m = len(P), len(Q)
    ca = -np.ones((n, m))

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(P[0] - Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i-1, 0), np.linalg.norm(P[i] - Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j-1), np.linalg.norm(P[0] - Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), np.linalg.norm(P[i] - Q[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return c(n-1, m-1)

from svgpathtools import svg2paths
from PIL import Image
import os
import pickle
import re
from utilsFCN import checkLimits, imagToPixel, checkPadding
import utilsCNN
import numpy as np
import matplotlib.pyplot as plt

debug = False

root = "dataset/"
dataset_padding = [0,0]
# file = 'Godzilla Racing R32'
# input = root+'input/'+file+'.png'
# output = root+'output/'+file+'.svg'

def checkPadding():
    directory = os.fsencode(root+"output/")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        output = root+'output/'+os.fsdecode(file)
        input = root+'input/'+filename+'.png'
        print(filename)

        paths, attributes = svg2paths(output)
        img = Image.open(input)
        width, height = img.size
        xlim = [paths[0][0].start.real,paths[0][0].start.real]
        ylim = [paths[0][0].start.imag,paths[0][0].start.imag]
        controls = []

        for path in paths:
            for curve in path:
                xlim = checkLimits(curve.start.real, xlim)
                ylim = checkLimits(curve.start.imag, ylim)
                xlim = checkLimits(curve.end.real, xlim)
                ylim = checkLimits(curve.end.imag, ylim)

        for path in paths:
            for curve in path:
                if(hasattr(curve, 'control1')):
                    controls.append(imagToPixel(curve.control1, height, width, xlim, ylim))
                if(hasattr(curve, 'control2')):
                    controls.append(imagToPixel(curve.control2, height, width, xlim, ylim))

        for control in controls:
            dataset_padding[0] = checkPadding(control[0], dataset_padding[0], width)
            dataset_padding[1] = checkPadding(control[1], dataset_padding[1], height)

        print(dataset_padding)

def checkCoordNum(window_size, dataset_sufix):
    directory = os.fsencode(root+"output/")
    max_coords = 0
    fat_file = ''
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        if not filename.endswith(f'_{dataset_sufix}'):
            continue
        output = root+'output/'+os.fsdecode(file)
        input = root+'input/'+filename+'.png'
        print(filename)

        paths, attributes = svg2paths(output)
        img = Image.open(input)
        width, height = img.size
        coord_num = []
        for i in range((width // window_size[0]) + 1):
            coord_num.append([])
            for j in range((height // window_size[1]) + 1):
                coord_num[i].append(0)
        xlim = [paths[0][0].start.real,paths[0][0].start.real]
        ylim = [paths[0][0].start.imag,paths[0][0].start.imag]

        for path in paths:
            for curve in path:
                xlim = checkLimits(curve.start.real, xlim)
                ylim = checkLimits(curve.start.imag, ylim)
                xlim = checkLimits(curve.end.real, xlim)
                ylim = checkLimits(curve.end.imag, ylim)

        for path in paths:
            for curve in path:
                x, y = imagToPixel(curve.start, height, width, xlim, ylim)
                x = x // window_size[0]
                y = y // window_size[1]
                coord_num[x][y] += 1
                
                if hasattr(curve, 'control1'):
                    x, y = imagToPixel(curve.control1, height, width, xlim, ylim)
                    x = x // window_size[0]
                    y = y // window_size[1]
                    coord_num[x][y] += 1
                if hasattr(curve, 'control2'):
                    x, y = imagToPixel(curve.control2, height, width, xlim, ylim)
                    x = x // window_size[0]
                    y = y // window_size[1]
                    coord_num[x][y] += 1
        
        for w in coord_num:
            for h in w:
                if h > max_coords:
                    max_coords = h
                    fat_file = filename
        
        if debug:
            break
        
    print(max_coords)
    print(fat_file)

def cleanStrokes():
    directory = os.fsencode(root+"output/")
    stroke_files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        output = root+'output/'+os.fsdecode(file)
        input = root+'input/'+filename+'.png'
        print(filename)
        delete = False

        with open(output, "r", encoding="utf-8") as f:
            for line in f:
                x = re.findall(r"stroke", line)
                if(x):
                    stroke_files.append(filename)
                    print('Se borrará :P')
                    if not debug:
                        delete = True
                    break

        if delete:
            os.remove(output)
            os.remove(input)
        
    with open(root+'stroke_files.pkl', 'wb') as pkl:
        pickle.dump(stroke_files, pkl)

def coordHistogram(dataset_sufix):
    directory = os.fsencode(root+"output/")
    histogram = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        regex = re.search(r'_fig_[0-9]+', filename)
        if regex == None:
            continue
        output = root+'output/'+os.fsdecode(file)

        paths, attributes = svg2paths(output)
        coord_num = 0
        for path in paths:
            for curve in path:
                coord_num += 1                
                if hasattr(curve, 'control1'):
                    coord_num += 1
                if hasattr(curve, 'control2'):
                    coord_num += 1
        
        if coord_num <= 30:
            histogram.append(coord_num)
        
        if debug:
            break
    
    print(len(histogram))
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(histogram)
    ax2.boxplot(histogram)
    plt.show()

def getCoordNumAllImage():
    directory = os.fsencode(root+'output/')
    max_coord = 0
    fat_file = ''
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        regex = re.search(r'_fig_[0-9]+', filename)
        if regex == None:
            continue
        output = f'{root}output/{filename}.svg'
        paths, attributes = svg2paths(output)
        coord_num = 0
        for path in paths:
            for curve in path:
                coord_num += 1
                if hasattr(curve, 'control1'):
                    coord_num += 1
                if hasattr(curve, 'control2'):
                    coord_num += 1
        
        if coord_num > max_coord:
            max_coord = coord_num
            fat_file = filename
            print(max_coord)
            print(fat_file)
    
    print(max_coord)
    print(fat_file)

def deleteFatFiles(coord_limit):
    directory = os.fsencode(root+'output/')
    fat_files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)[:-4]
        regex = re.search(r'_fig_[0-9]+', filename)
        if regex == None:
            continue
        output = f'{root}output/{filename}.svg'
        paths, attributes = svg2paths(output)
        coord_num = 0
        for path in paths:
            for curve in path:
                coord_num += 1
                if hasattr(curve, 'control1'):
                    coord_num += 1
                if hasattr(curve, 'control2'):
                    coord_num += 1
        
        if coord_num > coord_limit:
            fat_files.append(filename)
    
    for file in fat_files:
        print(f'Se borrará {file}')
        input = f'{root}input/{file}.png'
        output = f'{root}output/{file}.svg'
        if not debug:
            os.remove(input)
            os.remove(output)
    print(f'Se borraron {len(fat_files)} archivos')

def validate_blank(filename, shape, axis, dtype):
    input_matrix = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    blank_indices = np.where(np.all(input_matrix == 0, axis=axis))[0]
    print(f'El {filename} tiene ceros en:', blank_indices)

if __name__ == '__main__':
    coordHistogram(0)
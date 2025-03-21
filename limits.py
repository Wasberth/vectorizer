from svgpathtools import svg2paths
from PIL import Image
import os

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

dataset_padding = [0,0]

root = "C:/Users/sonic/Documents/USB/Escolar/TT/vectorizer/dataset/"
# file = 'Godzilla Racing R32'
# input = root+'input/'+file+'.png'
# output = root+'output/'+file+'.svg'

directory = os.fsencode(root+"output/")
for file in os.listdir(directory):
    output = root+'output/'+os.fsdecode(file)
    filename = os.fsdecode(file)[:-4]
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
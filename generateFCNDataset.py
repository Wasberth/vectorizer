from svgpathtools import svg2paths
from PIL import Image
import numpy as np
import re
import os
import multiprocessing

input_directory = "D:/Escolar/DatasetFCN/inputFCN/"
output_directory = "D:/Escolar/DatasetFCN/outputFCN/"
file_num = 0
root = 'C:/Users/sonic/Documents/USB/Escolar/TT/vectorizer/dataset/'
padding = 3
color_keyword_dict = {"aliceblue": "#F0F8FF",
    "antiquewhite": "#FAEBD7",
    "aqua": "#00FFFF",
    "aquamarine": "#7FFFD4",
    "azure": "#F0FFFF",
    "beige": "#F5F5DC",
    "bisque": "#FFE4C4",
    "black": "#000000",
    "blanchedalmond": "#FFEBCD",
    "blue": "#0000FF",
    "blueviolet": "#8A2BE2",
    "brown": "#A52A2A",
    "burlywood": "#DEB887",
    "cadetblue": "#5F9EA0",
    "chartreuse": "#7FFF00",
    "chocolate": "#D2691E",
    "coral": "#FF7F50",
    "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC",
    "crimson": "#DC143C",
    "cyan": "#00FFFF",
    "darkblue": "#00008B",
    "darkcyan": "#008B8B",
    "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9",
    "darkgreen": "#006400",
    "darkgrey": "#A9A9A9",
    "darkkhaki": "#BDB76B",
    "darkmagenta": "#8B008B",
    "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00",
    "darkorchid": "#9932CC",
    "darkred": "#8B0000",
    "darksalmon": "#E9967A",
    "darkseagreen": "#8FBC8F",
    "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F",
    "darkslategrey": "#2F4F4F",
    "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3",
    "deeppink": "#FF1493",
    "deepskyblue": "#00BFFF",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1E90FF",
    "firebrick": "#B22222",
    "floralwhite": "#FFFAF0",
    "forestgreen": "#228B22",
    "fuchsia": "#FF00FF",
    "gainsboro": "#DCDCDC",
    "ghostwhite": "#F8F8FF",
    "gold": "#FFD700",
    "goldenrod": "#DAA520",
    "gray": "#808080",
    "grey": "#808080",
    "green": "#008000",
    "greenyellow": "#ADFF2F",
    "honeydew": "#F0FFF0",
    "hotpink": "#FF69B4",
    "indianred": "#CD5C5C",
    "indigo": "#4B0082",
    "ivory": "#FFFFF0",
    "khaki": "#F0E68C",
    "lavender": "#E6E6FA",
    "lavenderblush": "#FFF0F5",
    "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD",
    "lightblue": "#ADD8E6",
    "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF",
    "lightgoldenrodyellow": "#FAFAD2",
    "lightgray": "#D3D3D3",
    "lightgreen": "#90EE90",
    "lightgrey": "#D3D3D3",
    "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A",
    "lightseagreen": "#20B2AA",
    "lightskyblue": "#87CEFA",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0",
    "lime": "#00FF00",
    "limegreen": "#32CD32",
    "linen": "#FAF0E6",
    "magenta": "#FF00FF",
    "maroon": "#800000",
    "mediumaquamarine": "#66CDAA",
    "mediumblue": "#0000CD",
    "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB",
    "mediumseagreen": "#3CB371",
    "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A",
    "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585",
    "midnightblue": "#191970",
    "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1",
    "moccasin": "#FFE4B5",
    "navajowhite": "#FFDEAD",
    "navy": "#000080",
    "oldlace": "#FDF5E6",
    "olive": "#808000",
    "olivedrab": "#6B8E23",
    "orange": "#FFA500",
    "orangered": "#FF4500",
    "orchid": "#DA70D6",
    "palegoldenrod": "#EEE8AA",
    "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE",
    "palevioletred": "#DB7093",
    "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9",
    "peru": "#CD853F",
    "pink": "#FFC0CB",
    "plum": "#DDA0DD",
    "powderblue": "#B0E0E6",
    "purple": "#800080",
    "red": "#FF0000",
    "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1",
    "saddlebrown": "#8B4513",
    "salmon": "#FA8072",
    "sandybrown": "#F4A460",
    "seagreen": "#2E8B57",
    "seashell": "#2E8B57",
    "sienna": "#A0522D",
    "silver": "#C0C0C0",
    "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#FFFAFA",
    "springgreen": "#00FF7F",
    "steelblue": "#4682B4",
    "tan": "#D2B48C",
    "teal": "#008080",
    "thistle": "#D8BFD8",
    "tomato": "#FF6347",
    "turquoise": "#40E0D0",
    "violet": "#EE82EE",
    "wheat": "#F5DEB3",
    "white": "#FFFFFF",
    "whitesmoke": "#F5F5F5",
    "yellow": "#FFFF00",
    "yellowgreen": "#9ACD32"
    }

def parseColor(color_str):
    if color_str.startswith("#"):
        color_str = color_str.lstrip("#")
        if len(color_str) == 3:  # Expandir colores como #FFF a #FFFFFF
            color_str = "".join(c*2 for c in color_str)
        return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
    elif color_str.startswith("rgb"):
        return tuple(map(int, re.findall(r"\d+", color_str)))  # Extraer valores RGB
    return (-1, -1, -1)  # Negro por defecto si no se encuentra un color válido

def compareColors(color1, color2):
    if(color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2]):
        return True
    return False

def concatMatrix(idx, input, output, lock):
    print(output)
    fcn_input = []
    fcn_output = []
    paths, attributes = svg2paths(output)
    color_classes = {}

    with open(output, "r", encoding="utf-8") as f:
        for line in f:
            x = re.findall("\.fil[0-9]+", line)
            if(x):
                key = x[0][1:]
                y = re.findall(r"\#[0-9A-F]{6}", line)
                if(y):
                    color_classes[key] = parseColor(y[0])
                else:
                    y = re.search(r"{fill:(\w+)", line)
                    if(y and y.group(1) != "none"):
                        color_classes[key] = parseColor(color_keyword_dict[y.group(1)])
                    else:
                        print("Wrong color class format")
                        print(line)

    img_pil = Image.open(input)
    width, heigth = img_pil.size

    x_se = {}
    y_se= {}
    x_control ={}
    y_control ={}
    original_arrays = {}
    output_arrays_se = {}
    output_arrays_control = {}

    for color in color_classes.keys():
        x_se[color] = []
        y_se[color] = []
        x_control[color] = []
        y_control[color] = []
        original_arrays[color] = np.zeros((heigth, width))
        output_arrays_se[color] = np.pad(original_arrays[color], pad_width=3, mode='constant', constant_values=0)
        output_arrays_control[color] = np.pad(original_arrays[color], pad_width=3, mode='constant', constant_values=0)

    x_lim = [paths[0][0].start.real, paths[0][0].start.real]
    y_lim = [0-paths[0][0].start.real, 0-paths[0][0].start.imag]
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
            x_se[usefull_class].append(curve.start.real)
            y_se[usefull_class].append(0-curve.start.imag)
            x_se[usefull_class].append(curve.end.real)
            y_se[usefull_class].append(0-curve.end.imag)

            if(curve.start.real > x_lim[1]):
                x_lim[1] = curve.start.real
            if(curve.end.real > x_lim[1]):
                x_lim[1] = curve.end.real
            if(0-curve.start.imag < y_lim[0]):
                y_lim[0] = 0-curve.start.imag
            if(0-curve.end.imag < y_lim[0]):
                y_lim[0] = 0-curve.end.imag
                
            if(curve.start.real < x_lim[0]):
                x_lim[0] = curve.start.real
            if(curve.end.real < x_lim[0]):
                x_lim[0] = curve.end.real
            if(0-curve.start.imag > y_lim[1]):
                y_lim[1] = 0-curve.start.imag
            if(0-curve.end.imag > y_lim[1]):
                y_lim[1] = 0-curve.end.imag

            if(hasattr(curve, 'control1')):
                x_control[usefull_class].append(curve.control1.real)
                y_control[usefull_class].append(0-curve.control1.imag)
            if(hasattr(curve, 'control2')):
                x_control[usefull_class].append(curve.control2.real)
                y_control[usefull_class].append(0-curve.control2.imag)

    img = np.array(img_pil)
    img[0][0] = img[0][1]

    for i in range(heigth):
        for j in range(width):
            for key, color in color_classes.items():
                if(compareColors(img[i][j], color)):
                    original_arrays[key][i][j] = 1

    for key in color_classes.keys():
        for i in range(len(x_se[key])):
            x = int((x_se[key][i]-x_lim[0])*(width-1)/(x_lim[1]-x_lim[0]))
            y = int((0-y_se[key][i]+y_lim[1])*(heigth-1)/(y_lim[1]-y_lim[0]))
            output_arrays_se[key][y+padding][x+padding] = 1

        for i in range(len(x_control[key])):
            x = int((x_control[key][i]-x_lim[0])*(width-1)/(x_lim[1]-x_lim[0]))
            y = int((0-y_control[key][i]+y_lim[1])*(heigth-1)/(y_lim[1]-y_lim[0]))
            output_arrays_control[key][y+padding][x+padding] = 1

    for key in color_classes.keys():
        fcn_input.append(original_arrays[key])
        fcn_output.append(np.stack((output_arrays_se[key], output_arrays_control[key]), axis=-1))

    np.save(input_directory+f"{idx}.npy", fcn_input)
    np.save(output_directory+f"{idx}.npy", fcn_output)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        available_threads = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(available_threads)

        thread_info = []
        i = 0

        directory = os.fsencode(root+"output/")
        for file in os.listdir(directory):
            filename = os.fsdecode(file)[:-4]
            input = root+'input/'+filename+'.png'
            output = root+'output/'+os.fsdecode(file)
            thread_info.append((i, input, output, lock))
            i += 1

        file_num = i

        print("Processing " + str(file_num) + " files")

        pool.starmap(concatMatrix, thread_info)

        pool.close()
        pool.join()
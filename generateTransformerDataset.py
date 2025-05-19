from svgpathtools import svg2paths
from PIL import Image
import numpy as np
import re
import os
import multiprocessing
import sys
import pickle
from utilsCNN import compareColors, parseColor, get_valid_indices, create_from_indices, controls_on_line

root = 'dataset/'
input_directory = root+"inputTransformer/"
output_directory = root+"outputTransformer/"
file_num = 0
vector_num = 15
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

def saveCoords(coord_x, coord_y, matrix):
    used_coord = len(coord_x)
    for i in range(used_coord):
        matrix[i][0] = coord_x[i]
        matrix[i][1] = coord_y[i]
    return matrix

def concatMatrixTransformer(img_pil, paths, attributes, color_classes, lock):
    valid = True
    width, height = img_pil.size
    input_matrix = np.zeros((width, height, 1))
    output_matrix = np.full((vector_num*3, 2), -1.0, dtype=np.float64)

    x_coord = []
    y_coord = []

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
                x_coord.append(curve.start.real)
                y_coord.append(curve.start.imag)

                if(hasattr(curve, 'control1')):
                    x_coord.append(curve.control1.real)
                    y_coord.append(curve.control1.imag)
                    x_coord.append(curve.control2.real)
                    y_coord.append(curve.control2.imag)
                else:
                    tmp = np.zeros((2,2))
                    tmp[0][0] = curve.start.real
                    tmp[0][1] = curve.start.imag
                    tmp[1][0] = curve.end.real
                    tmp[1][1] = curve.end.imag
                    c1, c2 = controls_on_line(tmp)
                    x_coord.append(c1[0])
                    x_coord.append(c2[0])
                    y_coord.append(c1[1])
                    y_coord.append(c2[1])
    
    x_coord.append(-1.0)
    y_coord.append(-1.0)

    img = np.array(img_pil)
    img[0][0] = img[0][1]

    for i in range(height):
        for j in range(width):
            for key, color in color_classes.items():
                if(key != 'fil0' and compareColors(img[i][j], color)):
                    input_matrix[i][j][0] = 1

    if len(x_coord) == 0:
        valid = False
    
    max_x_value = max(x_coord)
    max_y_value = max(y_coord)
    factor = 1
    if max_x_value > max_y_value:
        factor = 1/max_x_value
    else:
        factor = 1/max_y_value
    for i in range(len(x_coord)):
        x_coord[i] = x_coord[i]*factor
        y_coord[i] = y_coord[i]*factor
    
    output_matrix = saveCoords(x_coord, y_coord, output_matrix)

    return input_matrix, output_matrix, valid

def cursorPrint(id, text):
    sys.stdout.write(f"\033[{id + 1};0H")
    sys.stdout.write(f"\033[K")
    sys.stdout.write(text)
    sys.stdout.flush()

def generateTransformerMatrix(idx, idn, input, output, input_shape, output_shape, batch_size, thread_id, lock):
    iterations = int(np.ceil(len(idx)/batch_size))
    start = 0
    end = batch_size
    for i in range(iterations):
        if i == iterations-1:
            processing_ids = idx[start:]
            processing_inputs = input[start:]
            processing_outputs = output[start:]
        else:
            processing_ids = idx[start:end]
            processing_inputs = input[start:end]
            processing_outputs = output[start:end]

        imgs = [None]*batch_size
        paths = [None]*batch_size
        attributes = [None]*batch_size
        color_classes = [None]*batch_size
        input_matrices = [None]*batch_size
        output_matrices = [None]*batch_size
        for j in range(len(processing_ids)):
            imgs[j] = Image.open(processing_inputs[j])
            paths[j], attributes[j] = svg2paths(processing_outputs[j])
            color_classes[j] = {}
            with open(processing_outputs[j], 'r', encoding='utf-8') as f:
                for line in f:
                    x = re.findall(r"\.fil[0-9]+", line)
                    if x:
                        key = x[0][1:]
                        y = re.findall(r"\#[0-9A-F]{6}", line)
                        if y:
                            color_classes[j][key] = parseColor(y[0])
                        else:
                            y = re.search(r"{fill:(\w+)", line)
                            if(y and y.group(1) != "none"):
                                color_classes[j][key] = parseColor(color_keyword_dict[y.group(1)])

        for j in range(len(processing_ids)):
            input_matrices[j], output_matrices[j], valid = concatMatrixTransformer(imgs[j], paths[j], attributes[j], color_classes[j],  lock)
            if not valid:
                cursorPrint(20+thread_id, f'{processing_inputs[j]} no tiene nada')

        with lock:
            save_memmap = np.memmap(f'{input_directory}{idn}.npy', dtype=np.uint8, mode='r+', shape=input_shape)
            for use_id in range(len(processing_ids)):
                save_memmap[processing_ids[use_id]] = input_matrices[use_id]
            save_memmap.flush()
            save_memmap = np.memmap(f'{output_directory}{idn}.npy', dtype=np.float64, mode='r+', shape=output_shape)
            for use_id in range(len(processing_ids)):
                save_memmap[processing_ids[use_id]] = output_matrices[use_id]
            save_memmap.flush()
        
        cursorPrint(thread_id, f"Proceso {thread_id}: {((start+len(processing_ids)+1)*100/len(idx))}% completado")

        start = end
        end += batch_size

# Debug
def focus_id(idx, inputs, outputs, available_threads, sus_id):
    temp_idx = [[] for x in range(available_threads)]
    temp_inputs = [[] for x in range(available_threads)]
    temp_outputs = [[] for x in range(available_threads)]
    for n in range(len(idx[sus_id])):
        temp_idx[n%available_threads].append(idx[sus_id][n])
        temp_inputs[n%available_threads].append(inputs[sus_id][n])
        temp_outputs[n%available_threads].append(outputs[sus_id][n])
    return temp_idx, temp_inputs, temp_outputs

if __name__ == "__main__":
    create_files = True
    batch_size = 32
    dataset_sufix = 0
    print("\033[2J")
    multiprocessing.freeze_support()

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        available_threads = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(available_threads)

        thread_info = []
        i = 0
        idx = []
        inputs = []
        outputs = []
        for thread in range(available_threads):
            idx.append([])
            inputs.append([])
            outputs.append([])

        width = 0
        height = 0
        directory = os.fsencode(root+"output/")
        file_num = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)[:-4]
            regex = re.search(r'_fig_[0-9]+', filename)
            if regex != None:
                img = Image.open(root+'input/'+filename+'.png')
                width, height = img.size
                break

        for file in os.listdir(directory):
            filename = os.fsdecode(file)[:-4]
            regex = re.search(r'_fig_[0-9]+', filename)
            if regex != None:
                output = root+'output/'+os.fsdecode(file)

                paths, attributes = svg2paths(output)
                coord_num = 0
                for path in paths:
                    for curve in path:
                        coord_num += 1
                
                if coord_num < vector_num:
                    idx[i%available_threads].append(i)
                    inputs[i%available_threads].append(root+'input/'+filename+'.png')
                    outputs[i%available_threads].append(root+'output/'+os.fsdecode(file))
                    i += 1
        
        # DEBUG
        # idx, inputs, outputs = focus_id(idx, inputs, outputs, available_threads, 0)
        # idx, inputs, outputs = focus_id(idx, inputs, outputs, available_threads, 0)

        file_num = i
        input_shape = (file_num, width, height, 1)
        output_shape = (file_num, vector_num*3, 2)

        for thread in range(available_threads):
            thread_info.append((idx[thread], dataset_sufix, inputs[thread], outputs[thread], input_shape, output_shape, batch_size, thread, lock))

        if create_files:
            np.memmap(input_directory+f"{dataset_sufix}.npy", dtype=np.uint8, mode="w+", shape=input_shape)
            np.memmap(output_directory+f"{dataset_sufix}.npy", dtype=np.float64, mode="w+", shape=output_shape)

        cursorPrint(available_threads, "Procesando " + str(input_shape[0]) + " matrices")
        pool.starmap(generateTransformerMatrix, thread_info)
        pool.close()
        pool.join()

    cursorPrint(available_threads+3, 'Limpiando registros vacios...')
    indices = get_valid_indices(input_directory+f"{dataset_sufix}.npy", input_shape, (1, 2, 3), np.uint8)
    if len(indices) < input_shape[0]:
        create_from_indices(input_directory+f"{dataset_sufix}.npy", input_shape, np.uint8, indices)
        create_from_indices(output_directory+f"{dataset_sufix}.npy", output_shape, np.float64, indices)
        input_shape = (len(indices), input_shape[1], input_shape[2], input_shape[3])
        output_shape = (len(indices), vector_num*3, 2)
        indices = get_valid_indices(output_directory+f"{dataset_sufix}_filtered.npy", output_shape, (1, 2), np.float64)
    else:
        indices = get_valid_indices(output_directory+f"{dataset_sufix}.npy", output_shape, (1, 2), np.float64)
    if len(indices) < input_shape[0]:
        create_from_indices(input_directory+f"{dataset_sufix}.npy", input_shape, np.uint8, indices)
        create_from_indices(output_directory+f"{dataset_sufix}.npy", output_shape, np.float64, indices)
        input_shape = (len(indices), input_shape[1], input_shape[2], input_shape[3])
        output_shape = (len(indices), vector_num*3, 2)
    
    shape_dict = {}
    shape_dict['input'] = input_shape
    shape_dict['output'] = output_shape
    with open(f'{root}base_file_shapes.pkl', mode='wb') as base_shape_file:
        pickle.dump(shape_dict, base_shape_file)

    cursorPrint(available_threads+4, 'Ha terminado la ejecución\n')
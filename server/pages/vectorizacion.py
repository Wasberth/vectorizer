from flask import render_template, url_for, session, redirect, request, send_file
from decos import route
import os
import numpy as np
import subprocess
from PIL import Image
import re
from dotenv import load_dotenv
import json
import cv2
from skimage.filters.rank import modal
from io import BytesIO
load_dotenv()

def vectorize_to_svg(image, save_path, filename):
    # Cargar la imagen en RGB
    width, height = image.size
    image_np = np.array(image)

    # Encontrar colores únicos
    unique_colors = np.unique(image_np.reshape(-1, 3), axis=0)
    image_np = image_np.reshape((width*height, 3))
    print(f"Colores únicos encontrados: {len(unique_colors)}")
    i = 0
    paths = []
    for color in unique_colors:
        # Crear máscara binaria para el color
        mask = np.ones((width*height, 1), np.uint8) * 255
        mask[np.all(image_np == color, axis=-1)] = 0
        mask = mask.reshape((height, width))
        temp = Image.fromarray(mask, mode='L')
        bmp_path = f'{save_path}\\{filename}_temp.bmp' 
        temp.save(bmp_path)
        
        # Convertir color a formato hexadecimal
        hex_color = '#%02x%02x%02x' % tuple(color)

        subprocess.run([os.path.join(os.path.dirname(__file__) + '/potrace', 'potrace.exe'), bmp_path, '-o', f'{save_path}\\{filename}.svg', '--svg', '--color', hex_color])
        figure_path = ''
        with open(f'{save_path}\\{filename}.svg', 'r') as svg_file:
            finding = True
            for line in svg_file:
                if finding:
                    color = re.findall(r'#[0-9a-f]{6}', line)
                    if len(color) > 0:
                        print(color)
                        finding = False
                else:
                    if line.startswith('<path'):
                        if figure_path != '':
                            figure_path = figure_path[:-3]
                            figure_path += f' fill="{hex_color}" transform="translate(0.000000,{height}.000000) scale(0.100000,-0.100000)"/>'
                            paths.append(figure_path)
                            figure_path = ''
                    if line.startswith('</g>'):
                        figure_path = figure_path[:-3]
                        figure_path += f' fill="{hex_color}" transform="translate(0.000000,{height}.000000) scale(0.100000,-0.100000)"/>'
                        paths.append(figure_path)
                        break
                    figure_path += line[:-1] + ' '

        i += 1
    
    with open(f'{save_path}\\{filename}.svg', 'w') as f:
        f.write('<?xml version="1.0" standalone="no"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}" version="1.0" preserveAspectRatio="xMidYMid meet">\n')
        for pa in paths:
            f.write(pa+'\n')
        f.write('</svg>')

def classify(imagen, centroides):
    # Cargar la imagen y convertirla en un array de píxeles
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    pixels = np.array(imagen)[:,:,0:3]

    pixels = cv2.bilateralFilter(pixels, d=5, sigmaColor=75, sigmaSpace=75)

    # Convertir los colores de la imagen de RGB a LAB
    pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)

    cluster_centers = np.array(centroides, dtype=np.float32)
    diff = pixels_color_space[:, :, np.newaxis, :] - cluster_centers[np.newaxis, np.newaxis, :, :]
    distancias = np.linalg.norm(diff, axis=-1)

    indices = np.argmin(distancias, axis=-1)
    indices_modal = modal(indices.reshape(pixels.shape[0:2]), np.ones((3,3)))
    classified_image = cluster_centers[indices_modal, :].astype(np.uint8)
    
    imagen_segmentada_rgb = cv2.cvtColor(classified_image, cv2.COLOR_LAB2RGB)

    return Image.fromarray(imagen_segmentada_rgb)
    

@route('/vectorize/<filename>', methods=['POST'])
def vectorize(filename):
    centroides_request = json.loads(request.data)
    file_path = os.path.join(os.path.dirname(__file__) + '/uploaded', filename)
    centroides = []
    for centroide in centroides_request:
        centroides.append(centroide['centroide'])
    imagen = Image.open(file_path)
    imagen = classify(imagen, centroides)
    vectorize_to_svg(imagen, os.path.dirname(__file__) + '/uploaded', filename)
    return json.dumps({'estado': 'exito', 'siguiente': url_for('show_vector', filename=filename)})

@route('/vector/<filename>')
def show_vector(filename):
    return render_template(f'vector.html', stylesheets=['bootstrap.min'], scripts=['bootstrap.min'], filename=filename)

@route('/download/<filename>')
def descargar_svg(filename):
    file_path = os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename)
    # Probablemente aquí va validación
    svg_string = ''
    with open(file_path, 'r') as f:
        for line in f:
            svg_string += line
    buffer = BytesIO()
    buffer.write(svg_string.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='vectorized_img.svg')

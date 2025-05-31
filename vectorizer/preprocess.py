import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from skimage.filters.rank import modal
from sklearn.metrics import silhouette_score
import subprocess
import os
import re

def preprocess(imagen, max_k=30, model=None):
    """
    Preprocesa una imagen para su segmentación.
    
    Args:
        imagen (Image): Imagen a preprocesar.
        max_k (int, optional): Número máximo de clusters a utilizar. Valor por defecto es 30.
        
    Returns:
        ndarray: Matriz de pixeles de la imagen preprocesada.
    """

    # Cargar la imagen y convertirla en un array de píxeles
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    pixels = np.array(imagen)[:,:,0:3]

    pixels = cv2.bilateralFilter(pixels, d=5, sigmaColor=75, sigmaSpace=75)

    # Convertir los colores de la imagen de RGB a LAB
    pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
    pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista 2D de colores LAB

    if model==None:
        # Determinar el número óptimo de clusters usando el silhouette coefficient
        silhouette_scores = []  # Lista para almacenar los valores del silhouette score
        sample_size = len(pixels_2d)

        # Log values:
        # 
        # 100 empieza a reducir en 650 pixeles
        # 300 empieza a redicor en 2325 pixeles 
        # 587.05 empieza a reducir en 5000 pixeles
        # 1085.73 Empieza a reducir en 10000 pixeles
        # 

        sample_size = int(min(sample_size, np.ceil(500 * np.log(sample_size))))

        # # Evaluar silhouette coefficient para diferentes valores de k
        # for k in range(2, max_k + 1):  # El silhouette coefficient requiere al menos 2 clusters
        #     kmeans = KMeans(n_clusters=k, random_state=22)
        #     labels = kmeans.fit_predict(pixels_2d)
        #     score = silhouette_score(pixels_2d, labels, sample_size=sample_size)  # Calcular silhouette coefficient
        #     silhouette_scores.append(score)
        # 
        # # Determinar el número óptimo de clusters basado en el silhouette coefficient máximo
        # optimal_k = np.argmax(silhouette_scores) + 2  # Ajustar el índice (k inicia en 2)
        optimal_k = 3

        # Recolorear la imagen con los colores de los clusters obtenidos con el número óptimo de clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=22)
        kmeans.fit(pixels_2d)
        etiquetas = kmeans.labels_
    else:
        etiquetas = model.predict(pixels_2d)
        kmeans = model
    colores_clusters = kmeans.cluster_centers_

    # Reconstruir la imagen segmentada
    pixels_recolored = colores_clusters[etiquetas].reshape(pixels.shape).astype(np.uint8)

    # Convertir a RGB para OpenCV
    imagen_segmentada_rgb = cv2.cvtColor(pixels_recolored, cv2.COLOR_LAB2RGB)
    imagen_segmentada_rgb = modal(imagen_segmentada_rgb, np.ones((3,3,1)))

    # Convertir la imagen procesadada de nuevo a PIL y retornarla
    return Image.fromarray(imagen_segmentada_rgb), kmeans

def vectorize(image, save_path):
    # Cargar la imagen en RGB
    width, height = image.size
    image_np = np.array(image)

    # Encontrar colores únicos
    unique_colors = np.unique(image_np.reshape(-1, 3), axis=0)
    print(f"Colores únicos encontrados: {len(unique_colors)}")
    i = 0
    paths = []
    for color in unique_colors:
        # Crear máscara binaria para el color
        mask = np.all(image_np != color, axis=-1).astype(np.uint8) * 255
        temp = Image.fromarray(mask, mode='L')
        bmp_path = base_path+'\\temp.bmp' 
        temp.save(bmp_path)
        
        # Convertir color a formato hexadecimal
        hex_color = '#%02x%02x%02x' % tuple(color)

        subprocess.run(["C:/Users/sonic/Downloads/temp/potrace-1.16.win64/potrace.exe", bmp_path, '-o', f'{base_path}\\output_{i}.svg', '--svg', '--color', hex_color])
        figure_path = ''
        with open(f'{base_path}\\output_{i}.svg', 'r') as svg_file:
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
                            figure_path += f' fill="{hex_color}" transform="translate(0.000000,1184.000000) scale(0.100000,-0.100000)"/>'
                            paths.append(figure_path)
                            figure_path = ''
                    if line.startswith('</g>'):
                        figure_path = figure_path[:-3]
                        figure_path += f' fill="{hex_color}" transform="translate(0.000000,1184.000000) scale(0.100000,-0.100000)"/>'
                        paths.append(figure_path)
                        break
                    figure_path += line[:-1] + ' '

        i += 1
    
    with open(f'{base_path}\\output.svg', 'w') as f:
        f.write('<?xml version="1.0" standalone="no"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}" version="1.0" preserveAspectRatio="xMidYMid meet">\n')
        for pa in paths:
            f.write(pa+'\n')
        f.write('</svg>')

if __name__ == '__main__':
    path = 'C:/Users/sonic/3D Objects/input.png'
    base_path = os.path.dirname(__file__)
    imagen = Image.open(path)
    h, w = imagen.size
    total_size = h*w
    imagen, model = preprocess(imagen)
    imagen.save(base_path+'\\input.png', 'PNG')
    path = base_path+'\\input.png'
    if total_size < 300000:
        subprocess.run(['C:/Users/sonic/3D Objects/RESRGAN/realesrgan-ncnn-vulkan', '-i', path, '-o', f'{base_path}\\output.png', '-v', '1'])
        imagen = Image.open(base_path+'\\output.png')
        imagen, model = preprocess(imagen, model=model)
    imagen.save(base_path+'\\output.png', 'PNG')
    #imagen.show()
    vectorize(imagen, base_path)
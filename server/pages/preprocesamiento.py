from flask import render_template, url_for, session, redirect, request, send_from_directory
from decos import route
import os
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image
from skimage.filters.rank import modal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import subprocess
load_dotenv()
    
def preprocess(imagen, max_k=30, model=None, exact_k=0):
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

    if exact_k != 0:
        kmeans = KMeans(n_clusters=exact_k, random_state=22)
        kmeans.fit(pixels_2d)
        etiquetas = kmeans.labels_
    else:
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

            # Evaluar silhouette coefficient para diferentes valores de k
            for k in range(2, max_k + 1):  # El silhouette coefficient requiere al menos 2 clusters
                kmeans = KMeans(n_clusters=k, random_state=22)
                labels = kmeans.fit_predict(pixels_2d)
                score = silhouette_score(pixels_2d, labels, sample_size=sample_size)  # Calcular silhouette coefficient
                silhouette_scores.append(score)

            # Determinar el número óptimo de clusters basado en el silhouette coefficient máximo
            optimal_k = np.argmax(silhouette_scores) + 2  # Ajustar el índice (k inicia en 2)

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
    return Image.fromarray(imagen_segmentada_rgb), kmeans, pixels_color_space

def get_lab(imagen):
    # Cargar la imagen y convertirla en un array de píxeles
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    pixels = np.array(imagen)[:,:,0:3]

    pixels = cv2.bilateralFilter(pixels, d=5, sigmaColor=75, sigmaSpace=75)

    # Convertir los colores de la imagen de RGB a LAB
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)    

@route('/preprocesamiento/<name>')
def preprocesamiento(name):
    return render_template('preprocesamiento.html', stylesheets=['bootstrap.min', 'loading'], scripts=['bootstrap.min', 'image_loader'], filename=name)

@route('/uploaded/<filename>')
def descargar_imagen(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__) + os.environ['upload_path']), filename)

@route('/kmeans/<filename>', methods=['POST'])
def imagen_kmeans(filename):
    imagen_path = os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename)
    imagen = Image.open(imagen_path)
    imagen_procesada, model, pixel_color_space = preprocess(imagen)
    w, h = imagen.size
    pixel_count = h*w
    if pixel_count < int(os.environ['min_pixels']):
        imagen_procesada.save(os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename))
        return json.dumps({'estado':'SR', 'centroides': model.cluster_centers_.tolist(), 'siguiente': url_for('imagen_sr', filename=filename)})
    else:
        return json.dumps({'estado': 'exito', 'centroides': model.cluster_centers_.tolist(), 'pixels': pixel_color_space.tolist(), 'width': w, 'height': h})
    
@route('/super_resolution/<filename>', methods=['POST'])
def imagen_sr(filename):
    subprocess.run([os.path.join(os.path.dirname(__file__) + '/RESRGAN', 'realesrgan-ncnn-vulkan'), '-i', os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename), '-o', os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename)])
    return json.dumps({'estado': 'exito', 'siguiente': url_for('kmeans_SR', filename=filename)})

@route('/kmeans_sr/<filename>', methods=['POST'])
def kmeans_SR(filename):
    imagen_path = os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename)
    imagen = Image.open(imagen_path)
    w, h = imagen.size
    pixel_color_space = get_lab(imagen)
    return json.dumps({'estado': 'exito', 'pixels': pixel_color_space.tolist(), 'width': w, 'height': h})

@route('/cambiar_colores/<filename>', methods=['POST'])
def cambiar_colores(filename):
    datos = json.loads(request.data)
    imagen_path = os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], 'original_'+filename)
    imagen = Image.open(imagen_path)
    print('as')
    imagen_procesada, model, pixel_color_space = preprocess(imagen, exact_k=int(datos['numero']))
    w, h = imagen.size
    pixel_count = h*w
    if pixel_count < int(os.environ['min_pixels']):
        imagen_procesada.save(os.path.join(os.path.dirname(__file__) + os.environ['upload_path'], filename))
        return json.dumps({'estado':'SR', 'centroides': model.cluster_centers_.tolist(), 'siguiente': url_for('imagen_sr', filename=filename)})
    else:
        return json.dumps({'estado': 'exito', 'centroides': model.cluster_centers_.tolist(), 'pixels': pixel_color_space.tolist(), 'width': w, 'height': h})
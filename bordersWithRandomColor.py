import cv2
import numpy as np
from PIL import Image
import random

def detectar_bordes(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    # Aplicar el detector de bordes Canny
    bordes = cv2.Canny(gris, 100, 200)
    return bordes

def segmentar_areas(bordes):
    # Encontrar los componentes conectados en los bordes
    num_labels, labels_im = cv2.connectedComponents(bordes)
    return labels_im, num_labels

def colorear_areas(imagen, labels_im, num_labels):
    # Crear una imagen de salida para colorear las áreas segmentadas
    salida = np.zeros_like(imagen)
    for label in range(1, num_labels):  # Empieza en 1 para evitar el fondo
        # Asignar un color aleatorio a cada componente
        color = [random.randint(0, 255) for _ in range(3)]
        salida[labels_im == label] = color
    return salida

# Cargar la imagen
imagen = Image.open('christine.jpg')
imagen_np = np.array(imagen)

# Detectar bordes
bordes = detectar_bordes(imagen_np)

# Segmentar áreas
labels_im, num_labels = segmentar_areas(bordes)

# Colorear áreas
imagen_coloreada = colorear_areas(imagen_np, labels_im, num_labels)

# Convertir de vuelta a imagen y guardar/mostrar
bordes_final = Image.fromarray(bordes)
bordes_final.save('bordes6.png')
bordes_final.show()

imagen_final = Image.fromarray(imagen_coloreada)
imagen_final.save('imagen_random_coloreada6.png')
imagen_final.show()

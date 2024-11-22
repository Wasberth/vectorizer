import cv2
import numpy as np
from PIL import Image

def convertir_lab(imagen):
    """
    Convierte una imagen RGB al espacio de color LAB.
    
    :param imagen: Imagen de entrada (numpy array, RGB)
    :return: Imagen en espacio LAB
    """
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2Lab)

def aumentar_contraste_lab(imagen_lab):
    """
    Aumenta el contraste en el canal L (luminosidad) de una imagen en LAB.
    
    :param imagen_lab: Imagen en espacio LAB (numpy array)
    :return: Imagen LAB con contraste aumentado
    """
    l, a, b = cv2.split(imagen_lab)
    l_ecualizada = cv2.equalizeHist(l)
    return cv2.merge((l_ecualizada, a, b))

def quitar_antialiasing_umbralizacion(imagen_lab, umbral=128):
    """
    Aplica umbralización binaria en el canal L para quitar antialiasing.
    
    :param imagen_lab: Imagen en espacio LAB (numpy array)
    :param umbral: Umbral para binarización
    :return: Imagen binarizada (escala de grises)
    """
    l, _, _ = cv2.split(imagen_lab)
    _, binarizada = cv2.threshold(l, umbral, 255, cv2.THRESH_BINARY)
    return binarizada

def quitar_antialiasing_escalar(imagen, factor=2):
    """
    Escala una imagen hacia abajo y luego hacia arriba con interpolación "nearest".
    
    :param imagen: Imagen de entrada (numpy array, escala de grises)
    :param factor: Factor de reducción de tamaño
    :return: Imagen sin antialiasing
    """
    altura, ancho = imagen.shape[:2]
    nueva_dim = (ancho // factor, altura // factor)
    
    # Reducir tamaño
    reducida = cv2.resize(imagen, nueva_dim, interpolation=cv2.INTER_NEAREST)
    
    # Escalar de vuelta
    escalada = cv2.resize(reducida, (ancho, altura), interpolation=cv2.INTER_NEAREST)
    return escalada

# Script principal
if __name__ == "__main__":
    # Cargar imagen con PIL y convertir a numpy array
    imagen_path = 'img/test2.jpg'  # Cambia esta ruta a tu imagen
    imagen = Image.open(imagen_path)
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    imagen = np.array(imagen)
    
    # Convertir al espacio LAB
    #imagen = convertir_lab(imagen)
    
    # Aumentar contraste
    #imagen = aumentar_contraste_lab(imagen)
    
    # Aplicar umbralización
    #imagen = quitar_antialiasing_umbralizacion(imagen)
    
    # Aplicar escala para quitar antialiasing
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    imagen = quitar_antialiasing_escalar(imagen)
    
    # Guardar y mostrar resultados
    
    resultado_final = Image.fromarray(imagen)
    resultado_final.save('resultado_final.png')
    resultado_final.show()

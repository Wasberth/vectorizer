import numpy as np
from skfuzzy.cluster import cmeans
from PIL import Image
import cv2

def centroides_demasiado_cercanos(centroides, umbral=0.1):
    for i in range(len(centroides)):
        for j in range(i+1, len(centroides)):
            #print(np.linalg.norm(centroides[i] - centroides[j]))
            if np.linalg.norm(centroides[i] - centroides[j]) < umbral:
                return True
    return False

def calcular_distancia_media(sse):
    # Calcula la distancia de cada punto a la línea formada por el primer y último punto
    p1 = np.array([1, sse[0]])
    p2 = np.array([len(sse), sse[-1]])
    
    distancias = []
    for i in range(len(sse)):
        p = np.array([i + 1, sse[i]])
        distancia = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distancias.append(distancia)
    
    return distancias

# Cargar la imagen y convertirla en un array de píxeles
imagen = Image.open('a.jpg')
pixels = np.array(imagen)

pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista de colores RGB

# Determinar el número óptimo de clusters
max_k = 6  # Ajusta este valor según tus necesidades
optimal_k = 1
mfpc = 0
optimal_k = 0

for k in range(2, max_k + 1):
    cntr, u, u0, d, jm, p, fpc = cmeans(np.transpose(pixels_2d), k, m=3, error=0.005, maxiter=1000)
    if optimal_k < fpc:
        mfpc = fpc
        optimal_k = k

    print(fpc)

    if centroides_demasiado_cercanos(cntr):
        break

print(optimal_k)

cntr, u, u0, d, jm, p, fpc = cmeans(np.transpose(pixels_2d), optimal_k, m=3, error=0.005, maxiter=1000)
etiquetas = np.argmax(u, axis=0)
print(etiquetas)
print(np.array(cntr).astype(np.uint8))
print(np.transpose(pixels_2d))

# Recolorear la imagen con los colores de los clusters
colores_clusters = cntr

imagen_recolorada_color_space = np.array(colores_clusters[etiquetas]).reshape(pixels.shape).astype(np.uint8)
imagen_recolorada_rgb = cv2.cvtColor(imagen_recolorada_color_space, cv2.COLOR_LAB2RGB)
imagen_final = Image.fromarray(imagen_recolorada_rgb)

#imagen_recolorada = colores_clusters[etiquetas].reshape(pixels.shape).astype(np.uint8)
#imagen_final = Image.fromarray(imagen_recolorada)

imagen_final.save('imagen_recolorada.png')
imagen_final.show()


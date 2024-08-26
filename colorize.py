import numpy as np
from sklearn.cluster import KMeans
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
imagen = Image.open('christine.jpg')
pixels = np.array(imagen)

print(pixels)

pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista de colores RGB

#pixels_2d = pixels.reshape(-1, 3)  # Convertir a una lista de colores RGB

# Determinar el número óptimo de clusters
max_k = 30  # Ajusta este valor según tus necesidades
optimal_k = 1
sse = []

for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(pixels_2d)
    sse.append(kmeans.inertia_)

    #print(k)

    if centroides_demasiado_cercanos(kmeans.cluster_centers_):
        break

distancias = calcular_distancia_media(sse)
optimal_k = distancias.index(max(distancias)) + 1  # +1 porque los índices empiezan en 0
print(optimal_k)

# Recolorear la imagen con los colores de los clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=22)
kmeans.fit(pixels_2d)
colores_clusters = kmeans.cluster_centers_
etiquetas = kmeans.labels_

imagen_recolorada_color_space = colores_clusters[etiquetas].reshape(pixels.shape).astype(np.uint8)
imagen_recolorada_rgb = cv2.cvtColor(imagen_recolorada_color_space, cv2.COLOR_LAB2RGB)
imagen_final = Image.fromarray(imagen_recolorada_rgb)

#imagen_recolorada = colores_clusters[etiquetas].reshape(pixels.shape).astype(np.uint8)
#imagen_final = Image.fromarray(imagen_recolorada)

imagen_final.save('imagen_recolorada.png')
imagen_final.show()

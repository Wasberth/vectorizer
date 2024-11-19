import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def calcular_distancia_media(sse):
    """
    Calcula la distancia de cada punto a la línea formada por el primer y último punto de SSE (Suma de Errores Cuadrados)
    Usado para identificar el "codo" en la gráfica de SSE.
    """
    p1 = np.array([1, sse[0]])  # Primer punto
    p2 = np.array([len(sse), sse[-1]])  # Último punto
    
    distancias = []
    for i in range(len(sse)):
        p = np.array([i + 1, sse[i]])  # Puntos intermedios
        # Calcula la distancia perpendicular de cada punto a la línea p1-p2
        distancia = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distancias.append(distancia)
    
    return distancias

# Cargar la imagen y convertirla en un array de píxeles
imagen = Image.open('img/test2.jpg')
pixels = np.array(imagen)

# Convertir los colores de la imagen de RGB a LAB
pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista 2D de colores LAB

# Determinar el número óptimo de clusters usando el método del codo con una condición de parada
max_k = 30
sse = []  # Lista para almacenar la suma de los errores cuadrados para cada valor de k
threshold = 1e-10  # Umbral de cambio mínimo en SSE para detener el bucle

# Ejecutar KMeans para diferentes valores de k y almacenar el SSE
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(pixels_2d)
    sse.append(kmeans.inertia_)  # SSE para cada k

    # Comprobar si el cambio relativo en SSE es menor que el umbral
    if k > 1:
        cambio_relativo = abs(sse[-2] - sse[-1])
        if cambio_relativo < threshold:
            print(f"Deteniendo el bucle en k={k} debido a un cambio relativo bajo: {cambio_relativo:.4f}")
            break

print(sse)

# Calcular las distancias a la línea entre el primer y último punto de la curva SSE
distancias = calcular_distancia_media(sse)
#optimal_k = distancias.index(max(distancias)) + 1  # El valor de k donde la distancia es máxima
optimal_k = k-1

print(f"El número óptimo de clusters es: {optimal_k}")

# Opcional: Mostrar la gráfica del codo para visualizar el SSE y el punto óptimo
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(sse) + 1), sse, marker='o')
plt.title('Método del Codo para selección de K')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Suma de los Errores Cuadráticos (SSE)')
plt.axvline(x=optimal_k, linestyle='--', color='r', label=f'Optimal k = {optimal_k}')
plt.legend()
plt.show()

# Recolorear la imagen con los colores de los clusters obtenidos con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=22)
kmeans.fit(pixels_2d)
colores_clusters = kmeans.cluster_centers_
etiquetas = kmeans.labels_

# Convertir los colores de vuelta de LAB a RGB y reconstruir la imagen
imagen_recolorada_color_space = colores_clusters[etiquetas].reshape(pixels.shape).astype(np.uint8)
imagen_recolorada_rgb = cv2.cvtColor(imagen_recolorada_color_space, cv2.COLOR_LAB2RGB)
imagen_final = Image.fromarray(imagen_recolorada_rgb)

# Guardar y mostrar la imagen final
imagen_final.save('imagen_recolorada.png')
imagen_final.show()
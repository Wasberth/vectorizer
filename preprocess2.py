import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen y convertirla en un array de píxeles
imagen = Image.open('img/test2.jpg')
pixels = np.array(imagen)

# Convertir los colores de la imagen de RGB a LAB
pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista 2D de colores LAB

# Determinar el número óptimo de clusters usando el silhouette coefficient
max_k = 30
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
print("Evaluando")
for k in range(2, max_k + 1):  # El silhouette coefficient requiere al menos 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=22)
    labels = kmeans.fit_predict(pixels_2d)
    score = silhouette_score(pixels_2d, labels, sample_size=sample_size)  # Calcular silhouette coefficient
    print(k, score)
    silhouette_scores.append(score)

# Determinar el número óptimo de clusters basado en el silhouette coefficient máximo
optimal_k = np.argmax(silhouette_scores) + 2  # Ajustar el índice (k inicia en 2)

print(f"El número óptimo de clusters es: {optimal_k}")

# Mostrar la gráfica del silhouette coefficient
plt.figure(figsize=(8, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.title('Silhouette Coefficient para selección de K')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Coefficient')
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

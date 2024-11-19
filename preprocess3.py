import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2

# Cargar la imagen y convertirla en un array de píxeles
imagen = Image.open('img/test2.jpg')
pixels = np.array(imagen)

# Convertir los colores de la imagen de RGB a LAB
pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista 2D de colores LAB

# Ejecutar K-Means para agrupar los colores
num_clusters = 9
kmeans = KMeans(n_clusters=num_clusters, random_state=22)
labels = kmeans.fit_predict(pixels_2d)
colores_clusters = kmeans.cluster_centers_

# Convertir etiquetas a la forma original de la imagen
labels_image = labels.reshape(pixels.shape[:2])

# Crear representación one-hot encoding para las etiquetas
one_hot = np.eye(num_clusters)[labels_image]

# Aplicar operaciones morfológicas por color
erode_kernel = np.array(
    [
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ]
).astype(np.uint8)
dilate_kernel_1 = np.array(
    [
        [1,1,1], 
        [1,1,1], 
        [1,1,1]
    ]
).astype(np.uint8)
one_hot_processed = np.zeros_like(one_hot)  # Matriz para almacenar los resultados

for i in range(num_clusters):
    # Operaciones morfológicas en cada canal (cluster)
    channel = (one_hot[:, :, i] * 255).astype(np.uint8)  # Escalar a 0-255 para cv2
    channel = cv2.erode(channel, erode_kernel, iterations=1)
    channel = cv2.dilate(channel, dilate_kernel_1, iterations=1)
    channel = cv2.dilate(channel, erode_kernel, iterations=1)
    one_hot_processed[:, :, i] = channel / 255  # Normalizar de vuelta a 0-1

# Reconstruir etiquetas a partir del one-hot encoding procesado
labels_processed = np.argmax(one_hot_processed, axis=-1)

# Mapear etiquetas procesadas de vuelta a colores
pixels_recolored = colores_clusters[labels_processed.flatten()].reshape(pixels.shape).astype(np.uint8)

# Convertir de LAB a RGB
imagen_recolorada_rgb = cv2.cvtColor(pixels_recolored, cv2.COLOR_LAB2RGB)
imagen_final = Image.fromarray(imagen_recolorada_rgb)

# Guardar y mostrar la imagen final
imagen_final.save('imagen_recolorada_onehot_procesada.png')
imagen_final.show()

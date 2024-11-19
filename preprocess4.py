import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2

# Cargar la imagen y convertirla en un array de píxeles
imagen = Image.open('img/test3.png')
pixels = np.array(imagen)[:,:,0:3]
print(pixels.shape)

pixels = cv2.bilateralFilter(pixels, d=5, sigmaColor=75, sigmaSpace=75)

# Convertir los colores de la imagen de RGB a LAB
pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista 2D de colores LAB

# Aplicar K-Means
kmeans = KMeans(n_clusters=6, random_state=22)
labels = kmeans.fit_predict(pixels_2d)
colores_clusters = kmeans.cluster_centers_

# Reconstruir la imagen segmentada
pixels_recolored = colores_clusters[labels].reshape(pixels.shape).astype(np.uint8)

# Convertir a RGB para OpenCV
imagen_segmentada_rgb = cv2.cvtColor(pixels_recolored, cv2.COLOR_LAB2RGB)

# Convertir la imagen procesada de nuevo a PIL y guardar
imagen_final = Image.fromarray(imagen_segmentada_rgb)
imagen_final.save('imagen_pre_denoised.png')
imagen_final.show()

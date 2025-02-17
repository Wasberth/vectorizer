import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from skimage.filters.rank import modal
from sklearn.metrics import silhouette_score

def preprocess(imagen, max_k=30):
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
    colores_clusters = kmeans.cluster_centers_
    etiquetas = kmeans.labels_

    # Reconstruir la imagen segmentada
    pixels_recolored = colores_clusters[etiquetas].reshape(pixels.shape).astype(np.uint8)

    # Convertir a RGB para OpenCV
    imagen_segmentada_rgb = cv2.cvtColor(pixels_recolored, cv2.COLOR_LAB2RGB)
    imagen_segmentada_rgb = modal(imagen_segmentada_rgb, np.ones((3,3,1)))

    # Convertir la imagen procesada de nuevo a PIL y guardar
    #imagen_final = Image.fromarray(imagen_segmentada_rgb)
    #imagen_final.save('imagen_pre_denoised.png')
    #imagen_final.show()

    # Convertir la imagen procesadada de nuevo a PIL y retornarla
    return Image.fromarray(imagen_segmentada_rgb)

if __name__ == '__main__':
    imagen = Image.open('img/test4.png')
    imagen = preprocess(imagen)
    imagen.show()
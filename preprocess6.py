import numpy as np
import warnings
np.warnings = warnings
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.filters.rank import modal

def resize_to_max_mpx(image, max_mpx=3):
    """
    Redimensiona una imagen para que su resolución no exceda el tamaño máximo de megapíxeles especificado.
    
    :param image: Imagen de entrada en formato numpy array.
    :param max_mpx: Tamaño máximo en megapíxeles.
    :return: Imagen redimensionada.
    """
    # Obtén las dimensiones actuales de la imagen
    height, width = image.shape[:2]
    current_mpx = (height * width) / 1_000_000  # Calcula el tamaño actual en megapíxeles

    # Si la imagen ya está por debajo del tamaño máximo, regresa la original
    if current_mpx <= max_mpx:
        return image

    # Calcula el factor de escala necesario
    scale_factor = (max_mpx / current_mpx) ** 0.5

    # Calcula las nuevas dimensiones
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    print(f"resized to {new_width} x {new_height} = {new_height * new_width}")

    # Redimensiona la imagen
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

# Cargar la imagen y convertirla en un array de píxeles
imagen = Image.open('img/test4.png')
if imagen.mode != 'RGB':
    imagen = imagen.convert('RGB')
pixels = resize_to_max_mpx(np.array(imagen))

pixels = cv2.bilateralFilter(pixels, d=5, sigmaColor=75, sigmaSpace=75)

# Convertir los colores de la imagen de RGB a LAB
pixels_color_space = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
pixels_2d = pixels_color_space.reshape(-1, 3)  # Convertir a una lista 2D de colores LAB

amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(pixels_2d, amount_initial_centers).initialize()

# Crear y ejecutar el modelo X-Means
xmeans_instance = xmeans(pixels_2d, initial_centers, max_clusters=20)
xmeans_instance.process()

# Obtener etiquetas y centros
etiquetas = xmeans_instance.get_clusters()
colores_clusters = xmeans_instance.get_centers()

print(f"La cantidad de colores es {len(colores_clusters)}")

# Asignar el color de cada píxel al centro correspondiente
# Inicializamos una imagen vacía de la misma forma que la original
pixels_recolor = np.zeros_like(pixels_2d)

# Asignar el color de los centros según las etiquetas
for i, cluster in enumerate(etiquetas):
    for pixel_idx in cluster:
        pixels_recolor[pixel_idx] = colores_clusters[i]

# Convertir los colores de vuelta de LAB a RGB
pixels_recolor_rgb = cv2.cvtColor(pixels_recolor.reshape(pixels.shape), cv2.COLOR_LAB2RGB)

# Crear la imagen final
imagen_final = Image.fromarray(pixels_recolor_rgb)

# Guardar y mostrar la imagen final
imagen_final.save('imagen_recolorada-x.png')
imagen_final.show()

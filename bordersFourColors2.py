import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from skimage.filters.rank import modal


def apply_sobel(image):
    """
    Aplica el operador Sobel para detectar bordes en la imagen.
    
    :param image: Imagen de entrada en escala de grises.
    :return: Imagen binaria de bordes.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    _, sobel_binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    return sobel_binary.astype(np.uint8)


def segment_and_color(image):
    """
    Segmenta la imagen basada en contornos y asigna colores usando el teorema de los cuatro colores.
    
    :param image: Imagen binaria de bordes.
    :return: Imagen coloreada y etiquetas segmentadas.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_image = np.zeros(image.shape, dtype=np.int32)
    color_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Colores para el teorema de los 4 colores.

    # Asignación de colores a las regiones segmentadas
    for idx, contour in enumerate(contours):
        cv2.drawContours(labeled_image, [contour], -1, idx + 1, thickness=cv2.FILLED)

    adjacency_list = find_adjacency(labeled_image)

    # Coloreado basado en el teorema de 4 colores
    for label, neighbors in adjacency_list.items():
        used_colors = {colors[n - 1] for n in neighbors if n > 0}
        available_colors = [c for c in colors if c not in used_colors]
        color = available_colors[0] if available_colors else (255, 255, 255)
        color_image[labeled_image == label] = color

    return color_image, labeled_image


def find_adjacency(labeled_image):
    """
    Encuentra relaciones de vecindad entre etiquetas segmentadas.
    
    :param labeled_image: Imagen etiquetada.
    :return: Diccionario de etiquetas con sus vecinos.
    """
    height, width = labeled_image.shape
    adjacency = {label: set() for label in np.unique(labeled_image) if label > 0}

    for y in range(height):
        for x in range(width):
            label = labeled_image[y, x]
            if label > 0:
                neighbors = labeled_image[max(0, y - 1):min(height, y + 2),
                                          max(0, x - 1):min(width, x + 2)].ravel()
                adjacency[label].update(neighbors)
                adjacency[label].discard(label)

    return adjacency

def plot_boxplot(labeled_image):
    """
    Genera un boxplot para la distribución de los tamaños de las regiones segmentadas.
    
    :param labeled_image: Imagen etiquetada.
    """
    unique_labels, counts = np.unique(labeled_image[labeled_image > 0], return_counts=True)
    plt.boxplot(counts, vert=False)
    plt.title("Distribución de tamaños de regiones segmentadas")
    plt.xlabel("Cantidad de píxeles")
    plt.show()

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


if __name__ == "__main__":
    # Carga la imagen
    input_image = cv2.imread("img/test2.jpg", cv2.IMREAD_GRAYSCALE)
    input_image = resize_to_max_mpx(input_image)

    # Aplicar el operador Sobel
    edges = apply_sobel(input_image)
    print("edges")

    # Segmentación y coloreado
    colored_image, labeled_image = segment_and_color(edges)
    print("colored")

    # Reducción de ruido
    #smoothed_image = modal(labeled_image, np.ones((3,3)))
    print("ruido")

    # Mostrar y guardar resultados
    #cv2.imshow("Segmentación coloreada", smoothed_image)
    #cv2.waitKey(0)

    # Graficar el boxplot
    #plot_boxplot(smoothed_image)

    cv2.imwrite("imagen_con_myuchas_operaciones.jpg", labeled_image)
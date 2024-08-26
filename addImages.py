import cv2

ruta1 = 'christine.jpg'
ruta2 = 'imagen_recolorada.png'
ruta3 = 'suma.jpg'

# Cargar las imágenes
image1 = cv2.imread(ruta1)
image2 = cv2.imread(ruta2)

# Asegurarse de que ambas imágenes tienen el mismo tamaño
if image1.shape != image2.shape:
    raise ValueError("Las imágenes deben tener el mismo tamaño y número de canales")

# Sumar las imágenes
sum_image = cv2.add(image1, image2)

# Guardar la imagen resultante
cv2.imwrite(ruta3, sum_image)

print(f"La imagen resultante ha sido guardada en {ruta3}")
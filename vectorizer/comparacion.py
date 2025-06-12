from PIL import Image
import numpy as np

def calculate_mae(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    arr1 = np.asarray(img1, dtype=np.float32)
    arr2 = np.asarray(img2, dtype=np.float32)

    mae = np.mean(np.abs(arr1 - arr2))
    return mae

imagen_base = 'C:/Users/sonic/3D Objects/Untitled.png'
imagen_propuesta = 'C:/Users/sonic/3D Objects/vectorized_img.png'
imagen_profesional = 'C:/Users/sonic/3D Objects/corel_detail_bone_shakers.png'

mae_value = calculate_mae(imagen_base, imagen_propuesta)
print('Métricas de nuestra propuesta')
print(f"MAE: {mae_value}")
percentage = 1-(mae_value/765)
print(f"Similitud: {percentage*100}%")
print()

mae_value = calculate_mae(imagen_base, imagen_profesional)
print('Métricas de CorelDRAW')
print(f"MAE: {mae_value}")
percentage = 1-(mae_value/765)
print(f"Similitud: {percentage*100}%")

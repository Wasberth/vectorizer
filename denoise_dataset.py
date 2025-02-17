import os 
from PIL import Image
import time

from vectorizer.preprocess import preprocess

INPUT_FOLDER = 'dataset/input'
DENOISED_FOLDER = 'dataset/denoised'

os.makedirs(DENOISED_FOLDER, exist_ok=True)

for file in os.listdir(INPUT_FOLDER):
#    if file.endswith('.png'):
    print(file)
    start = time.time()
    imagen = preprocess(Image.open(os.path.join(INPUT_FOLDER, file)))
    imagen.save(os.path.join(DENOISED_FOLDER, file))
    end = time.time()
    print(f'Took {end - start} seconds')

print('Done!')
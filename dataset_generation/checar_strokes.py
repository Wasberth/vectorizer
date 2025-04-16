import re
import os

directory = "D:\My Files\Catalogo\catalogo_svg"  # Ruta al directorio que contiene los archivos .svg

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    regex = r'str0'

    # Buscar la primera coincidencia
    match = re.search(regex, content)
    if match:
        print(filename)
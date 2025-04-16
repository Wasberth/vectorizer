import re
import os

directory = "D:\My Files\Catalogo\dataset2\catalogo_svg"  # Ruta al directorio que contiene los archivos .svg

for filename in os.listdir(directory):
    print(filename, end="\r")
    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    regex = r'<path class="(.*)" d="M-0.35 0.8c1181,0 2362,0 3543,0 0,1574.66 0,3149.33 0,4723.99 -1181,0 -2362,0 -3543,0 0,-1574.66 0,-3149.33 0,-4723.99z"/>'

    # Buscar la primera coincidencia
    match = re.search(regex, content)
    if match and match.group(1) != "fil0":
        print(filename)
        print(match.group(1))
        print()
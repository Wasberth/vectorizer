import os
import re
import json

CATALOGO_PATH = "D:\My Files\Catalogo\output"

for file in os.listdir(CATALOGO_PATH):
    print(file)
    if file.endswith('.svg'):
        file_path = os.path.join(CATALOGO_PATH, file)
        with open(file_path, 'r+') as f:
            content = f.read()
            content = re.sub(r'^\s*<line([^>]*)/>\s*$', '', content, flags=re.MULTILINE)
            f.seek(0)
            f.write(content)
            f.truncate()  # Elimina cualquier contenido sobrante si la nueva versión es más corta
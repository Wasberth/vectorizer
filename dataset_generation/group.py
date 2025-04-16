import os
import re
import json

CATALOGO_PATH = "D:\My Files\Catalogo\output"
grouped_files = {}

for file in os.listdir(CATALOGO_PATH):
    print(file)
    if file.endswith('.svg'):
        with open(os.path.join(CATALOGO_PATH, file), 'r') as f:
            content = f.read()
            match = re.findall(r'(?<=\<)([^\s>\/]+)(?=[^>]*\>)', content)
            tags = frozenset(match)
            if tags not in grouped_files:
                grouped_files[tags] = []
            grouped_files[tags].append(file)

dumpable = {}
mapped_tags = []
i = 0
for key, value in grouped_files.items():
    dumpable[i] = value
    mapped_tags.append(list(key))
    i += 1

# Guardar a un archivo
with open('D:/My Files/Catalogo/grouped.json', 'w') as f:
    json.dump(dumpable, f)

# Escribir el mapeo de tags a un archivo
with open('D:/My Files/Catalogo/tags.json', 'w') as f:
    json.dump(mapped_tags, f)




import os
import re
import json

CATALOGO_PATH = "D:\My Files\Catalogo\datasetp\catalogo_svg"
INPUT_PATH = "D:\My Files\Catalogo\datasetp\input"
OUTPUT_PATH = "D:\My Files\Catalogo\datasetp\output"

regex_class_def = r'\.(fil\d+) \{[^\}]*fill:none'
regex_class_inst = lambda clazz: rf'<path[^>]*\bclass="[^"]*\b{re.escape(clazz)}\b[^"]*"'

for file in os.listdir(CATALOGO_PATH):
    if file.endswith('.svg'):
        file_name = os.path.splitext(file)[0]
        suffixes = [0, 1, 2, 3, 4]

        file_path = os.path.join(CATALOGO_PATH, file)

        with open(file_path, 'r+') as f:
            content = f.read()

        # Find the class definitions
        has_instance = False

        class_defs = re.findall(regex_class_def, content)
        for class_def in class_defs:
            class_name = class_def.split()[0]

            # Find if at least one instance of the class is present
            first_instance = re.search(regex_class_inst(class_name), content)
            if first_instance:
                has_instance = True
                break

        if not has_instance:
            continue

        print('Deleting', file_name)

        # Delete the files
        for suffix in suffixes:
            input_path = os.path.join(INPUT_PATH, file_name + f'_{suffix}.png')
            output_path = os.path.join(OUTPUT_PATH, file_name + f'_{suffix}.svg')
            os.remove(input_path)
            os.remove(output_path)

        os.remove(file_path)
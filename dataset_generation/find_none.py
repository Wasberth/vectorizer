import os
import re
import json

CATALOGO_PATH = "D:\My Files\Catalogo\datasetp\catalogo_svg"
INPUT_PATH = "D:\My Files\Catalogo\datasetp\input"
OUTPUT_PATH = "D:\My Files\Catalogo\datasetp\output"

for file in os.listdir(CATALOGO_PATH):
    if file.endswith('.svg'):
        file_name = os.path.splitext(file)[0]
        suffixes = [0, 1, 2, 3, 4]

        file_path = os.path.join(CATALOGO_PATH, file)

        with open(file_path, 'r+') as f:
            content = f.read()
        # Check if the file has a <line> tag
        if re.search(r'^\s*<line([^>]*)/>\s*$', content, flags=re.MULTILINE):
            print(f"File {file} has a <line> tag")
        else:
            continue

        # Delete the files
        os.remove(file_path)

        for suffix in suffixes:
            input_path = os.path.join(INPUT_PATH, file_name +f'_{suffix}.png')
            output_path = os.path.join(OUTPUT_PATH, file_name +f'_{suffix}.svg')
            os.remove(input_path)
            os.remove(output_path)

            print(f"File {file} deleted")

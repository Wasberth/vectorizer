import os
import re
import json

INPUT_PATH = "D:\My Files\Catalogo\dataset2\input"
OUTPUT_PATH = "D:\My Files\Catalogo\dataset2\output"

print("Checking for missing files...")
for file in os.listdir(INPUT_PATH):
    if file.endswith('.png'):
        #print(f"Checking file {file}...")
        file_name = os.path.splitext(file)[0]

        # Check if the file exists in the output directory with png extension
        if not os.path.exists(os.path.join(OUTPUT_PATH, file_name + '.svg')):
            print(f"File {file} is missing!")
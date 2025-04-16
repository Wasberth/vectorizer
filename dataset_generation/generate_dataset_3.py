import os
import random
import subprocess
from PIL import Image
import numpy as np
from resize_svg import resize_svg
import re
import time

SVG_DIRECTORY = os.path.abspath("./catalogo_svg")
INPUT_DIRECTORY = os.path.abspath("./input")
OUTPUT_DIRECTORY = os.path.abspath("./output")
#CATALOGO_DIRECTORY = os.path.abspath("./catalogo")

def process_svgs(directory = SVG_DIRECTORY):
    """
    Procesa todos los archivos SVG en un directorio y genera archivos PNG y SVG modificados.

    :param directory: Ruta del directorio donde están los archivos SVG.
    """

    # Check if input and output directories exist
    input_exists = os.path.exists(INPUT_DIRECTORY)
    output_exists = os.path.exists(OUTPUT_DIRECTORY)
    if not input_exists and not output_exists:
        print("Directorios de entrada y salida no encontrados. Por favor, cree los directorios 'input' y 'output' en el directorio raíz del proyecto.")
        return

    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp.svg")
    temp_dir = os.path.dirname(temp_path)
    temp_file = os.path.basename(temp_path)

    path_png = lambda name, i : os.path.join(INPUT_DIRECTORY, f"{name}_{i}.png")
    path_svg = lambda name, i : os.path.join(OUTPUT_DIRECTORY, f"{name}_{i}.svg")

    # Delete temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    for file in os.listdir(directory):
        if file.endswith(".svg"):
            start = time.time()
            percentage = 0
            print(f"Procesando {file}. 0%", end="\r")
            og_svg_path = os.path.join(directory, file)

            base_name = os.path.splitext(file)[0]

            # 2. Elegir medidas aleatorias (entre 150 y 600 px)
            standard_x = 1080
            standard_y = 1440

            min_x = 420
            max_x = 810
            min_y = 560
            max_y = 1080

            random_antialias = 0

            dimensions = [(standard_x, standard_y), (min_x, min_y), (min_x, max_y),
                          (max_x, min_y), (max_x, max_y)]
            for i in range(len(dimensions)):
                if input_exists:
                    svg_to_png(directory, file, path_png(base_name, i), *dimensions[i], random_antialias)
                if output_exists:
                    rescale_svg(og_svg_path, path_svg(base_name, i), *dimensions[i])
                percentage += 20
                print(f"Procesando {file}. {percentage}%", end="\r")

            # Delete temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            end = time.time()
            print(f"Procesado: {file}. Tardó {end - start} segundos")

def svg_to_png(svg_dir, svg_file, png_path, width, height, antialias):
    """
    Convierte un archivo SVG a PNG usando Inkscape y aplica rotación.

    :param svg_path: Ruta del archivo SVG de entrada.
    :param png_path: Ruta del archivo PNG de salida.
    :param width: Ancho deseado.
    :param height: Alto deseado.
    :param rotation: Ángulo de rotación en grados.
    """

    try:
        subprocess.Popen([
            "inkscape",
            svg_file,
            "--export-type=png",
            f'--export-filename={png_path}',
            f"--export-width={width}",
            f"--export-height={height}",
            f"--export-png-antialias={antialias}",
        ], shell=True, cwd=svg_dir).wait()
    except Exception as e:
        print(f"Error al convertir {svg_file}: {e}")

    imagen = Image.open(png_path)
    pixels = np.array(imagen)[:,:,0:3]
    if pixels[0,0,0] == 255:
        pixels[0,0,0] = 254
    else:
        pixels[0,0,0] += 1
    
    imagen_final = Image.fromarray(pixels)
    imagen_final.save(png_path)

def rescale_svg(input_svg, output_svg, new_width, new_height):
    """
    Reescala un SVG correctamente ajustando la viewBox y todos los puntos dentro del SVG.

    :param input_svg: Ruta del archivo SVG de entrada.
    :param output_svg: Ruta del archivo SVG de salida.
    :param new_width: Nuevo ancho.
    :param new_height: Nuevo alto.
    :param rotation: Ángulo de rotación en grados.
    """

    with open(input_svg) as f:
        svg_string = f.read()
        svg_string = resize_svg(svg_string, new_width, new_height)
        with open(output_svg, "w") as f:
            f.write(svg_string)

def noisy_svg(svg_path, remove = 0.5):
    def delete(match: re.Match):
        """
        Elimina un elemento SVG con una probabilidad de `remove`.
        """
        if random.random() < remove:
            return ""
        return match.group(0)
    
    with open(svg_path) as f:
        svg_string = f.read()
        svg_string = re.sub(
            r'^\s*<path class=\"fil[1-9]\d*\"[^>]*>\s*$', 
            delete, svg_string, flags=re.MULTILINE
        )
        return svg_string

# Ejecutar el script
if __name__ == "__main__":
    process_svgs()

import os
import random
import subprocess
from PIL import Image
from xml.dom import minidom
import svgutils
import numpy as np
from resize_svg import resize_svg

SVG_DIRECTORY = os.path.abspath("./catalogo_svg")
INPUT_DIRECTORY = os.path.abspath("./input")
OUTPUT_DIRECTORY = os.path.abspath("./output")
CATALOGO_DIRECTORY = os.path.abspath("./catalogo")

def process_svgs(directory = SVG_DIRECTORY):
    """
    Procesa todos los archivos SVG en un directorio y genera archivos PNG y SVG modificados.

    :param directory: Ruta del directorio donde están los archivos SVG.
    """

    for file in os.listdir(directory):
        if file.endswith(".svg"):
            svg_path = os.path.join(directory, file)

            base_name = os.path.splitext(file)[0]

            # 1. Generar PNG de 300x400 en el catalogo
            png_fixed_path = os.path.join(CATALOGO_DIRECTORY, f"{base_name}.png")
            svg_to_png(directory, file, png_fixed_path, 300, 400, 2)

            # 2. Elegir medidas aleatorias (entre 150 y 600 px)
            random_width = random.randint(300, 1200)
            random_height = random.randint(300, 1200)
            random_antialias = 0

            # 4. Generar PNG aleatorio en input
            png_random_path = os.path.join(INPUT_DIRECTORY, f"{base_name}.png")
            svg_to_png(directory, file, png_random_path, random_width, random_height, random_antialias)

            # 5. Generar SVG con medidas aleatorias en output
            svg_random_path = os.path.join(OUTPUT_DIRECTORY, f"{base_name}.svg")
            #rescale_and_rotate_svg(svg_path, svg_random_path, random_width, random_height)
            rescale_svg(svg_path, svg_random_path, random_width, random_height)

            print(f"Procesado: {file}")

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

def transform_path(d, scale_x, scale_y):
    """
    Escala correctamente los comandos de un path SVG.

    :param d: Atributo "d" del path original.
    :param scale_x: Factor de escala en X.
    :param scale_y: Factor de escala en Y.
    :return: Nuevo atributo "d" escalado.
    """
    new_d = []
    tokens = d.split()
    for token in tokens:
        try:
            value = float(token)
            if len(new_d) % 2 == 0:
                new_d.append(str(value * scale_x))  # X
            else:
                new_d.append(str(value * scale_y))  # Y
        except ValueError:
            new_d.append(token)  # Mantener comandos (M, L, C, etc.)
    return " ".join(new_d)

# Ejecutar el script
if __name__ == "__main__":
    process_svgs()

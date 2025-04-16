import os
import random
import subprocess
from PIL import Image
import numpy as np
import re
import time
import multiprocessing

from resize_svg import resize_svg
from sample_svg import get_points_on_path_fast

SVG_ORIGIN_DIRECTORY = os.path.abspath("./catalogo_svg")
SHAPE_DIRECTORY = os.path.abspath("./shapes")
PNG_DEST_DIRECTORY = os.path.abspath("./input")
SVG_DEST_DIRECTORY = os.path.abspath("./output")
NUMBER_DEST_DIRECTORY = os.path.abspath("./samples")

sufixer = lambda i : f"_{i}" if i != "" else ""
path_png = lambda name, i = "" : os.path.join(PNG_DEST_DIRECTORY, f"{name}{sufixer(i)}.png")
path_svg = lambda name, i = "" : os.path.join(SVG_DEST_DIRECTORY, f"{name}{sufixer(i)}.svg")
path_numpy = lambda name, i = "" : os.path.join(NUMBER_DEST_DIRECTORY, f"{name}{sufixer(i)}.npy")

STANDARD_DIMENTIONS = (1080, 1440) #, (420, 560), (420, 1080), (810, 560), (810, 1080)]
STANDARD_ANTIALIAS = 0
STANDARD_SAMPLES = 1000

def process_svg(svg_dir, svg_file, sufix = None,
        dimensions = STANDARD_DIMENTIONS, antialias = STANDARD_ANTIALIAS,
        samples = STANDARD_SAMPLES, delete_resized_svg = True, delete_figure_svg = True):
    start = time.time()

    print(f"Procesando: {svg_file}.")
    base_name = os.path.splitext(svg_file)[0]
    svg_origin = os.path.join(svg_dir, svg_file)
    svg_dest = path_svg(base_name, sufix)
    figure_name = base_name + sufixer(sufix) + '_fig'

    rescale_svg(svg_origin, svg_dest, *dimensions)

    for i, figure in enumerate(separate_figures(svg_dest)):
        figure_text, figure_path = figure

        with open(path_svg(figure_name, i), "w") as f:
            f.write(figure_text)

        svg_to_png(SVG_DEST_DIRECTORY, figure_name + sufixer(i) + '.svg',
                   path_png(figure_name, i), *dimensions, antialias)

        if samples > 0:
            sampled_path = get_points_on_path_fast(figure_path, samples)
            np.save(path_numpy(figure_name, i), sampled_path)

    if delete_resized_svg:
        os.remove(svg_dest)

    if delete_figure_svg:
        for i in range(len(dimensions)):
            os.remove(path_svg(figure_name, i))

    end = time.time()
    print(f"Procesado: {svg_file}. Tardó {end - start} segundos")

def process_svgs(directory = SVG_ORIGIN_DIRECTORY):
    """
    Procesa todos los archivos SVG en un directorio y genera archivos PNG y SVG modificados.

    :param directory: Ruta del directorio donde están los archivos SVG.
    """
    multiprocessing.freeze_support()

    with multiprocessing.Manager() as manager:
        available_processes = multiprocessing.cpu_count()
        print(f"Available processes: {available_processes}")
        pool = multiprocessing.Pool(processes=available_processes)

        thread_info = []

        for file in os.listdir(directory):
            if file.endswith(".svg"):
                thread_info.append((directory, file))
        
        pool.starmap(process_svg, thread_info)
        pool.close()
        pool.join()


def separate_figures(svg_path):
    svg_template = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Creator: CorelDRAW 2021.5 -->
<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" width="1080px" height="1440px" version="1.1" style="shape-rendering:geometricPrecision; text-rendering:geometricPrecision; image-rendering:optimizeQuality; fill-rule:evenodd; clip-rule:evenodd"
viewBox="0 0 1080 1440"
 xmlns:xlink="http://www.w3.org/1999/xlink"
 xmlns:xodm="http://www.corel.com/coreldraw/odm/2003">
 <defs>
  <style type="text/css">
   <![CDATA[
    .fil0 {fill:black}
    .fil1 {fill:white}
   ]]>
  </style>
 </defs>
 <g id="Layer_x0020_1">
  <metadata id="CorelCorpID_0Corel-Layer"/>
  <path class="fil0" d="%s"/>
  <g id="_2589601767600">
   <path class="fil1" d="%s"/>
  </g>
 </g>
</svg>
"""

    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    fil0 = re.search(r"<path class=\"fil0\" d=\"([^\"]*)\"", content)
    if not fil0:
        print('WARNING: No fil0 found')
        return None, None

    fil0_dims = fil0.group(1)

    # Buscar todas las clases del tipo fill-<número>
    paths = re.findall(r"<path class=\"(fil\d+)\" d=\"([^\"]*)\"", content)

    for path in paths:
        if path[0] == "fil0":
            continue

        # Aplicar reemplazo de la clase actual
        yield svg_template % (fil0_dims, path[1]), path[1]

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
    directory = SVG_ORIGIN_DIRECTORY

    multiprocessing.freeze_support()

    with multiprocessing.Manager() as manager:
        available_processes = multiprocessing.cpu_count()
        print(f"Available processes: {available_processes}")
        pool = multiprocessing.Pool(processes=available_processes//2)

        thread_info = []

        for file in os.listdir(directory):
            if file.endswith(".svg"):
                thread_info.append((directory, file))
        
        pool.starmap(process_svg, thread_info)
        pool.close()
        pool.join()


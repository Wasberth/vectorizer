import os
import random
import subprocess
from PIL import Image
from xml.dom import minidom
import svgutils
import numpy as np

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
            random_width = random.randint(150, 600)
            random_height = random.randint(150, 600)
            random_antialias = random.randint(0, 3)

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

def svg_to_svg(input_dir, input_file, output_path, width, height):
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
            input_file,
            "--export-type=svg",
            f'--export-filename={output_path}',
            f"--export-width={width}",
            f"--export-height={height}",
        ], shell=True, cwd=input_dir).wait()
    except Exception as e:
        print(f"Error al convertir {input_file}: {e}")

def rescale_and_rotate_svg(input_svg, output_svg, width, height):
    """
    Reescala y rota un archivo SVG.

    :param input_svg: Ruta del archivo SVG de entrada.
    :param output_svg: Ruta del archivo SVG de salida.
    :param width: Ancho deseado.
    :param height: Alto deseado.
    :param rotation: Ángulo de rotación en grados.
    """
    doc = minidom.parse(input_svg)
    svg_elem = doc.documentElement

    svg_elem.setAttribute("width", f"{width}px")
    svg_elem.setAttribute("height", f"{height}px")

    # Aplicar rotación
    #transform_attr = f"rotate({rotation} {width/2} {height/2})"
    #existing_transform = svg_elem.getAttribute("transform")
    #svg_elem.setAttribute("transform", existing_transform + " " + transform_attr if existing_transform else transform_attr)

    with open(output_svg, "w") as f:
        f.write(doc.toxml())

def rescale_svg(input_svg, output_svg, width, height):
    """
    Reescala y rota un archivo SVG.

    :param input_svg: Ruta del archivo SVG de entrada.
    :param output_svg: Ruta del archivo SVG de salida.
    :param width: Ancho deseado.
    :param height: Alto deseado.
    :param rotation: Ángulo de rotación en grados.
    """
    svg = svgutils.transform.fromfile(input_svg)
    mm_width = float(svg.width[:-2])
    mm_height = float(svg.height[:-2])
    print(mm_width, mm_height)
    px_width = mm_width * 0.2645833333
    px_height = mm_height * 0.2645833333

    originalSVG = svgutils.compose.SVG(input_svg)
    originalSVG.scale(width / px_width, height / px_height)
    figure = svgutils.compose.Figure(width, height, originalSVG)
    figure.save(output_svg)

# Ejecutar el script
if __name__ == "__main__":
    process_svgs()

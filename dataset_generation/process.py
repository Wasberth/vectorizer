import os
import random
import subprocess

RANDOM_MIN = 150
RANDOM_MAX = 600

SVG_DIRECTORY = os.path.abspath("./catalogo_svg")
INPUT_DIRECTORY = os.path.abspath("./input")
OUTPUT_DIRECTORY = os.path.abspath("./output")
CATALOGO_DIRECTORY = os.path.abspath("./catalogo")

def list_svgs(directory):
    """
    Lista todos los archivos SVG en un directorio.

    :param directory: Ruta del directorio donde están los archivos SVG.
    :return: Lista de rutas de los archivos SVG.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".svg")]

def process_svgs(input_path, files, output_path, export_type = "png", **kwargs): 
    """
    Procesa todos los archivos SVG en un directorio y genera archivos del tipo export_type modificados.

    :param input_path: Ruta del directorio donde están los archivos SVG.
    :param files: Lista de nombres de los archivos SVG a procesar.
    :param output_path: Ruta del directorio donde se almacenarán los archivos modificados.
    :param width: Ancho deseado.
    :param height: Alto deseado.
    :param export_type: Tipo de exportación. Por defecto, "png".
    """

    kwargs.get("width", random.randint(RANDOM_MIN, RANDOM_MAX))
    kwargs.get("height", random.randint(RANDOM_MIN, RANDOM_MAX))

    subprocess.run([
        "inkscape",
        files.join(' '),
        f"--export-type={export_type}",
        f'--export-filename="{output_path}"',
        f"--export-width={kwargs.get("width")}",
        f"--export-height={kwargs.get("height")}",
    ], shell=True, cwd=os.path.dirname(input_path))
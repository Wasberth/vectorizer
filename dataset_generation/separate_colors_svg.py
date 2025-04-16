import re
import os

def replace(match, target_class):
    """
    Reemplaza la coincidencia si contiene la clase dada.

    :param match: Objeto Match de una etiqueta con atributo class
    :param target_class: Clase que se desea eliminar
    :return: Etiqueta original o vacía si contiene la clase objetivo
    """
    if match.group(1) == "fil0":
        return match.group(0)

    class_list = match.group(1).split()
    if target_class in class_list:
        return match.group(0)
    return ""

def replacer(target_class):
    """
    Devuelve una función que actúa como reemplazo en re.sub.

    :param target_class: Clase objetivo
    :return: Función que se pasa a re.sub
    """
    return lambda match: replace(match, target_class)

directory = "D:\My Files\Catalogo\catalogo_svg_com"  # Ruta al directorio que contiene los archivos .svg
save_directory = "D:\My Files\Catalogo\catalogo_svg"  # Ruta al directorio donde se guardarán los archivos .svg
number = 0

for filename in os.listdir(directory):
    print(filename)
    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Buscar todas las clases del tipo fill-<número>
    classes = set(re.findall(r"(?<=\.)fil\d+", content))

    for clazz in classes:
        if clazz == "fil0":
            continue
        # Crear una copia del contenido original
        new_content = content

        # Aplicar reemplazo de la clase actual
        pattern = r'<[^>]*class="([^\">]*)"[^>]*>'
        rep = replacer(clazz)
        new_content = re.sub(pattern, rep, new_content)

        # Guardar archivo con nombre indicando la clase eliminada
        name, ext = os.path.splitext(filename)
        output_filename = f"{number}.svg"
        number += 1
        output_path = os.path.join(save_directory, output_filename)

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(new_content)

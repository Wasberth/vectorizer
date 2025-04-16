import re
import os

svg_template = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Creator: CorelDRAW 2021.5 -->
<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" width="3543px" height="4724px" version="1.1" style="shape-rendering:geometricPrecision; text-rendering:geometricPrecision; image-rendering:optimizeQuality; fill-rule:evenodd; clip-rule:evenodd"
viewBox="0 0 3543 4724"
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
  <path class="fil0" d="M-0.35 0.8c1181,0 2362,0 3543,0 0,1574.66 0,3149.33 0,4723.99 -1181,0 -2362,0 -3543,0 0,-1574.66 0,-3149.33 0,-4723.99z"/>
  <g id="_2589601767600">
   <path class="fil1" d="%s"/>
  </g>
 </g>
</svg>
"""

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
    paths = re.findall(r"<path class=\"(fil\d+)\" d=\"([^\"]*)\"", content)

    for path in paths:
        if path[0] == "fil0":
            continue

        # Aplicar reemplazo de la clase actual
        pattern = r'%s'
        new_content = re.sub(pattern, path[1], svg_template)

        # Guardar archivo con nombre indicando la clase eliminada
        name, ext = os.path.splitext(filename)
        output_filename = f"{str(number).zfill(6)}.svg"
        print(number, end="\r")
        number += 1
        output_path = os.path.join(save_directory, output_filename)

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(new_content)
    print()

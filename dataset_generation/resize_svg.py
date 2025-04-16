"""
Resize a SVG file to a new size. Coordinates of paths and gradient definitions get transposed to corresponding
values in the new canvas. Everything else remain unchanged.
"""


import re
from argparse import ArgumentParser
from xml.etree import ElementTree
import numpy as np

_SVG_NAMESPACE = 'http://www.w3.org/2000/svg'
_VALID_PATH_COMMANDS = "MLTHVCSQAZ"
_RE_FLOAT = re.compile(r'[+-]?\d*(\.\d+)?')


def resize_path(path, x_factor, y_factor):
    result = ""
    index = 0
    length = len(path)

    def eat_number(factor):
        nonlocal result
        nonlocal index
        match = _RE_FLOAT.match(path[index:])
        if not match:
            return
        found = match.group(0)
        scaled = factor * float(found)
        index += len(found)
        result += f"{scaled:.4f}".rstrip('0').rstrip('.')

    def skip_space():
        nonlocal index
        while path[index] == " " or path[index] == ",":
            index += 1

    def eat_space():
        nonlocal result
        skip_space()
        result += " "

    def eat_scale_xy():
        nonlocal result
        eat_number(x_factor)
        skip_space()
        result += ","
        eat_number(y_factor)

    def eat_for_command(command):
        if command in "MLT":
            eat_scale_xy()
        elif command == "H":
            eat_number(x_factor)
        elif command == "V":
            eat_number(y_factor)
        elif command == "C":
            eat_scale_xy()
            eat_space()
            eat_scale_xy()
            eat_space()
            eat_scale_xy()
        elif command in "SQ":
            eat_scale_xy()
            eat_space()
            eat_scale_xy()
        elif command == "A":
            eat_scale_xy()
            eat_space()
            eat_number(1)  # x-axis-rotation
            eat_space()
            eat_number(1)  # large-arc-flag
            eat_space()
            eat_number(1)  # sweep-flag
            eat_space()
            eat_scale_xy()
        elif command == "Z":
            pass
        else:
            raise ValueError("Unknown command", command)

    repeating_command = ''
    while index < length:
        skip_space()
        lead = path[index].upper()
        if lead in _VALID_PATH_COMMANDS:
            result += path[index]
            index += 1
            eat_for_command(lead)
            repeating_command = lead
        else:
            result += " "
            eat_for_command(repeating_command)

    return result


def _resize_element_path(el, x_factor, y_factor):
    path = el.get('d')
    el.set('d', resize_path(path, x_factor, y_factor))


def _resize_element_svg(el, width, height):
    assert type(width) == str
    assert type(height) == str
    el.set('width', width + "px")
    el.set('height', height + "px")
    el.set('viewBox', f'0 0 {width} {height}')


def _resize_element_gradient(el, x_factor, y_factor):
    for attr in ['x1', 'y1', 'x2', 'y2']:
        value = el.get(attr)
        if value:
            factor = x_factor if attr.startswith('x') else y_factor
            new_value = float(value) * factor
            el.set(attr, str(new_value))


def resize_svg(source, width, height):
    "Resize source svg to `width` and `height`"
    source = polygon_to_path(source)
    ElementTree.register_namespace('', _SVG_NAMESPACE)
    x_factor = 1
    y_factor = 1
    root = ElementTree.fromstring(source)
    for element in root.iter():
        if element.tag.endswith('svg'):
            viewbox = element.get('viewBox').split(' ')
            _resize_element_svg(element, str(width), str(height))
            x_factor = 1.0 / float(viewbox[2]) * float(width)
            y_factor = 1.0 / float(viewbox[3]) * float(height)
        elif element.tag.endswith('Gradient'):  # (linear|radial)Gradient
            _resize_element_gradient(element, x_factor, y_factor)
        elif element.tag.endswith('path'):
            _resize_element_path(element, x_factor, y_factor)
    return ElementTree.tostring(root, encoding='unicode')

def polygon_to_path(svg_string):
    """
    Convierte todas las etiquetas <polygon> en un string SVG a <path> con curvas de Bézier cúbicas.
    
    :param svg_string: El contenido del archivo SVG como string.
    :return: El SVG modificado con <polygon> convertidos a <path>.
    """
    def bezier_points(p1, p2):
        """
        Calcula dos puntos de control intermedios entre dos puntos para una curva de Bézier cúbica.
        :param p1: Punto inicial (x1, y1)
        :param p2: Punto final (x2, y2)
        :return: Dos puntos intermedios (cx1, cy1), (cx2, cy2)
        """

        lerp = lambda a, b, t: a * (1 - t) + b * t

        p1, p2 = np.array(p1), np.array(p2)
        
        return lerp(p1, p2, 1/3), lerp(p1, p2, 2/3)
    
    def convert_polygon(match: re.Match):
        """
        Convierte un solo <polygon> a <path>.
        """
        bezier_curve = lambda c1, c2, point_rel: f"{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {point_rel[0]:.2f},{point_rel[1]:.2f}"

        points_str = match.group(2)
        points = np.array([list(map(float, p.split(','))) for p in points_str.strip().split()])
        points = points
        zero = np.array([0,0])

        if len(points) < 3:
            return match.group(0)  # No modificar si no es un polígono válido
        
        current_center = points[0]
        d = [f"M{current_center[0]} {current_center[1]}c"]
        for i in range(1, len(points)):
            point = points[i]
            point_rel = point - current_center
            c1, c2 = bezier_points(zero, point_rel)
            d.append(bezier_curve(c1, c2, point_rel) + " ")
            current_center = point
        # Cerrar el path volviendo al inicio con Bézier
        point_rel = points[0] - current_center
        c1, c2 = bezier_points(zero, points[0] - current_center)
        d.append(bezier_curve(c1, c2, point_rel))
        d.append("z")

        d_string = ''.join(d)
        
        return f'<path{match.group(1)}d="{d_string}"{match.group(3)}>'
    
    return re.sub(r'<polygon([^>]*?)points="([^"]+)"([^>]*?)>', convert_polygon, svg_string)


def main():
    arg_parser = ArgumentParser(description="Resize SVG files.")
    arg_parser.add_argument(
        'source',
        metavar='SOURCE_SVG',
        help='Original SVG to resize'
    )
    arg_parser.add_argument(
        'width',
        metavar='WIDTH',
        type=float,
        help='Width of the new SVG'
    )
    arg_parser.add_argument(
        'height',
        metavar='HEIGHT',
        type=float,
        help='Height of the new SVG'
    )
    args = arg_parser.parse_args()
    return resize_svg(open(args.source).read(), args.width, args.height)


if __name__ == "__main__":
    resized = main()

    with open('resized.svg', 'w') as f:
        f.write(resized)
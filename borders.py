from PIL import Image
import os
from collections import defaultdict

def fill_holes(image, color, background_color):
    """
    Rellena los huecos en una capa con el color que mejor se ajuste.
    """
    from PIL import ImageDraw

    # Crear una máscara para los huecos
    mask = Image.new("1", image.size, 0)  # 0 es negro (huecos)
    draw = ImageDraw.Draw(mask)

    # Dibujar el color actual en la máscara
    for x in range(image.width):
        for y in range(image.height):
            if image.getpixel((x, y)) == color:
                draw.point((x, y), 1)  # 1 es blanco (área del color)

    # Rellenar los huecos en la máscara
    from PIL import ImageOps
    mask = ImageOps.invert(mask)  # Invertir la máscara para que los huecos sean blancos
    mask = mask.convert("L")  # Convertir a escala de grises
    mask = mask.point(lambda p: p > 128 and 255)  # Umbralizar para obtener una máscara binaria

    # Rellenar los huecos con el color de fondo
    filled_image = image.copy()
    draw = ImageDraw.Draw(filled_image)
    draw.floodfill((0, 0), background_color, mask=mask)

    return filled_image

def create_stackable_layers(image_path):
    # Abrir la imagen
    image = Image.open(image_path).convert("RGB")
    pixels = image.load()

    # Crear la carpeta "stackable_layers" si no existe
    if not os.path.exists("stackable_layers"):
        os.makedirs("stackable_layers")

    # Obtener todos los colores únicos en la imagen
    unique_colors = set()
    for i in range(image.width):
        for j in range(image.height):
            unique_colors.add(pixels[i, j])

    # Ordenar los colores (el primer color será el fondo)
    sorted_colors = sorted(unique_colors)
    background_color = sorted_colors[0]

    # Crear una imagen de fondo (primera capa)
    background_layer = Image.new("RGB", image.size, background_color)
    background_layer.save("stackable_layers/layer_0_background.png")

    # Crear capas de recorte para los demás colores
    for idx, color in enumerate(sorted_colors[1:], start=1):
        # Crear una nueva imagen en blanco (transparente)
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer_pixels = layer.load()

        # Pintar de color sólido los píxeles que coinciden con el color actual
        for i in range(image.width):
            for j in range(image.height):
                if pixels[i, j] == color:
                    layer_pixels[i, j] = color + (255,)  # Añadir canal alpha (opaco)

        # Rellenar los huecos en la capa
        filled_layer = fill_holes(layer, color, background_color)

        # Guardar la capa de recorte
        filled_layer.save(f"stackable_layers/layer_{idx}_cutting.png")

    print(f"Se han creado {len(sorted_colors)} capas en la carpeta 'stackable_layers'.")

# Ejemplo de uso
create_stackable_layers("D:/My Files/Documentos/Codigos/ai-image-to-vector/layers.png")
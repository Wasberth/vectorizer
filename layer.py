from PIL import Image
import numpy as np
from collections import deque
import os

def flood_get_mask(image, color, x, y):
    """
    Performs a flood fill from a given starting point (x, y) and returns a mask of the filled region.

    :param image: np.ndarray - Input image (Height x Width x Channels)
    :param color: np.ndarray - The target color for flood filling
    :param x: int - Starting x-coordinate
    :param y: int - Starting y-coordinate
    :return: np.ndarray - A binary mask with filled pixels set to 1
    """
    height, width = image.shape[:2]  # Ensure height and width are correctly assigned
    mask = np.zeros((height, width), dtype=bool)

    queue = deque([(x, y)])
    visited = set([(x, y)])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        x, y = queue.popleft()
        mask[x, y] = 1  # Mark as visited

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < height and 0 <= ny < width 
                and np.array_equal(image[nx, ny], color) 
                and (nx, ny) not in visited):  # Avoid duplicates

                queue.append((nx, ny))
                visited.add((nx, ny))  # Mark as visited immediately

    return mask

def fill_holes(mask):
    """
    Identifies and fills holes in the given binary mask.
    
    :param mask: np.ndarray - Binary mask where 1 indicates filled regions.
    :return: np.ndarray - Mask with holes filled.
    """
    print("Filling Holes")
    height, width = mask.shape
    processed_pixels = mask.copy()
    visited = np.zeros_like(mask, dtype=bool)

    for x in range(height):
        for y in range(width):
            if processed_pixels[x, y] >= 1 or visited[x, y]:
                continue

            print(f"Checking hole at ({x}, {y})")
            hole_candidate = flood_get_mask(mask, np.array(0, dtype=mask.dtype), x, y)

            # Mark as visited to avoid redundant processing
            visited |= hole_candidate

            # Check if the hole touches a border
            if np.any(hole_candidate[0, :]) or np.any(hole_candidate[:, 0]) or \
               np.any(hole_candidate[-1, :]) or np.any(hole_candidate[:, -1]):
                continue

            mask |= hole_candidate  # Fill the hole

    return mask

def layer_image(image: Image):
    """
    Segments an image into layers based on color clustering using flood fill.
    
    :param image: Image - PIL image object.
    """
    pixels = np.array(image, dtype=np.uint8)[:, :, :3]

    bin_shape = (image.height, image.width)
    print(f"Processing image with shape: {bin_shape}")

    processed_pixels = np.zeros(bin_shape, dtype=bool)
    layer_mask = np.zeros(bin_shape, dtype=bool)
    current_layer = np.zeros_like(pixels)
    layer_count = 0

    os.makedirs("stackable_layers", exist_ok=True)

    while not np.all(processed_pixels):
        currently_processed = np.logical_or(processed_pixels, layer_mask)
        next_pixel = np.argwhere(currently_processed == False)
        x = next_pixel[0][0]
        y = next_pixel[0][1]

        print(f"Processing region at ({x}, {y})")
        color = pixels[x, y]
        current_color_mask = flood_get_mask(pixels, color, x, y)

        # Update masks efficiently
        processed_pixels |= current_color_mask
        print("Mask computed")

        filled_color_mask = fill_holes(current_color_mask)
        layer_mask |= filled_color_mask
        print("Holes filled")

        # Apply color to the current layer
        current_layer[filled_color_mask] = color

        # Save the current layer as an image
        layer_image = Image.fromarray(current_layer.astype(np.uint8))
        layer_image.save(f"stackable_layers/layer_{layer_count}.png")
        layer_count += 1

        if np.all(layer_mask):
            layer_mask.fill(False)  # Reset layer mask for next iteration
            layer_mask |= processed_pixels

if __name__ == '__main__':
    image = Image.open('layer_test.png')
    layer_image(image)

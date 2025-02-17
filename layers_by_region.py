from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import os

class Region:

    def __init__(self, image: np.ndarray, x: int, y: int, **kwargs):
        self._mask = kwargs.get('mask') or Region.flood_mask(image, x, y)
        self._filled_mask = kwargs.get('filled_mask') or None
        self._color = image[x, y]

        arg = np.argwhere(self._mask)
        min_x, min_y = arg.min(axis=0)  # Extraer los valores mínimos
        max_x, max_y = arg.max(axis=0) + 1  # Extraer los valores máximos y ajustar

        bounds = [min_x, min_y, max_x, max_y]
        print("Bounds:", bounds)

        self._bounds = bounds

    @staticmethod
    def flood_get_mask(image: np.ndarray, color, x, y):
        height, width = image.shape[:2]  # Ensure height and width are correctly assigned

        # Get a color not on the image
        image_colors = set(image[:, :])
        available_colors = [c for c in Region._COLORS if c not in image[x, y]]


    @staticmethod
    def flood_mask(image: np.ndarray, x, y):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        color = image[x, y]

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

    @property
    def mask(self):
        return self._mask

    @property
    def color(self):
        return self._color

    @property
    def bounds(self):
        return self._bounds

    @property
    def filled_mask(self):
        if self._filled_mask is not None:
            return self._filled_mask
        
        small_mask = self._mask[self.bounds[0]:self.bounds[2], self.bounds[1]:self.bounds[3]].copy()
        processed = small_mask.copy()
        holes = np.zeros_like(processed, dtype=bool)

        while not np.all(processed):
            next_pixel = np.argwhere(processed == False)
            x = next_pixel[0][0]
            y = next_pixel[0][1]

            hole_candidate = Region.flood_mask(small_mask, x, y)
            processed |= hole_candidate

            # If the hole touches a border, skip it
            if np.any(hole_candidate[0, :]) or np.any(hole_candidate[:, 0]) or \
               np.any(hole_candidate[-1, :]) or np.any(hole_candidate[:, -1]):
                continue

            # Apply color to the current layer
            holes |= hole_candidate

        self._filled_mask = self._mask.copy()
        self._filled_mask[self.bounds[0]:self.bounds[2], self.bounds[1]:self.bounds[3]] |= holes

        return self._filled_mask

def layer_image(image: Image):
    pixels = np.array(image, dtype=np.uint8)[:, :, :3]

    bin_shape = (image.height, image.width)
    print(f"Processing image with shape: {bin_shape}")

    processed_pixels = np.zeros(bin_shape, dtype=bool)
    #layer_mask = np.zeros(bin_shape, dtype=bool)
    #current_layer = np.zeros_like(pixels)
    #layer_count = 0

    os.makedirs("stackable_layers", exist_ok=True)

    while not np.all(processed_pixels):
        next_pixel = np.argwhere(processed_pixels == False)
        x = next_pixel[0][0]
        y = next_pixel[0][1]

        print(f"Processing region at ({x}, {y})")
        region = Region(pixels, x, y)
        processed_pixels |= region.mask
        
        # Save binary image of the region
        region_image = Image.fromarray(region.mask.astype(np.uint8) * 255)
        region_image.save(f"stackable_layers/layer_{x}_{y}.png")
        
        # Save binary image of the filled region
        filled_region_image = Image.fromarray(region.filled_mask.astype(np.uint8) * 255)
        filled_region_image.save(f"stackable_layers/layer_{x}_{y}_filled.png")


if __name__ == '__main__':
    image = Image.open('layer_test.png')
    layer_image(image)
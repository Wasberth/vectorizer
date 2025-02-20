import cv2
import numpy as np

# Load the image and convert it float32 (kmeans requires float32 type).
original_image = cv2.imread('layer_test.png')
image = original_image.astype(np.int64)

cols, rows = image.shape[1], image.shape[0]

# Reshape the image into a 2D array with one row per pixel and three columns for the color channels.
data = image.reshape((cols * rows, 3))
# Map all colors to a unique integer value.
data = (data[:, 0] << 16) + (data[:, 1] << 8) + data[:, 2]

colors = np.unique(data, axis=0)  # Get the unique colors in the image
print(colors)

i = 0

for color in colors:
    mask = np.zeros((rows*cols, 1), np.uint8)  # Create a zeroed mask in the size of image.
    mask[data == color] = 255  # Set all pixels with the same color to 255.
    mask = mask.reshape((rows, cols))  # Reshape the mask back to the size of the image.
    #cv2.imshow(f'mask {k}', mask)  # Show mask for testing

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # Find contours
    for c in cnts:
        colored_mask = np.zeros_like(original_image)  # Initialize colored_mask with zeros
        contour_mask = np.zeros_like(original_image)  # Initialize colored_mask with zeros
        x, y = tuple(c[0][0])  # First coordinate in the contour
        color = original_image[y, x]  # Get the color of the pixel in that coordinate
        cv2.drawContours(colored_mask, [c], 0, [255, 255, 255], -1)  # Draw contour with the specific color
        cv2.imwrite(f'layers/figure{i}.png', colored_mask)  # Save as PNG for testing

        cv2.drawContours(contour_mask, [c], 0, [255, 255, 255], 1)  # Draw contour with the specific color
        cv2.imwrite(f'layers/contour{i}.png', contour_mask)  # Save as PNG for testing
        i += 1
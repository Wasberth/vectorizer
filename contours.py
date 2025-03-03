import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert it float32 (kmeans requires float32 type).
original_image = cv2.imread('layer_test.png')
image = original_image.astype(np.int64)

cols, rows = image.shape[1], image.shape[0]

# Reshape the image into a 2D array with one row per pixel and three columns for the color channels.
data = image.reshape((cols * rows, 3))
# Map all colors to a unique integer value.
data = (data[:, 0] << 16) + (data[:, 1] << 8) + data[:, 2]

colors = np.unique(data, axis=0)  # Get the unique colors in the image

all_contours = np.zeros_like(original_image)

i = 0

for color in colors:
    mask = np.zeros((rows*cols, 1), np.uint8)  # Create a zeroed mask in the size of image.
    mask[data == color] = 255  # Set all pixels with the same color to 255.
    mask = mask.reshape((rows, cols))  # Reshape the mask back to the size of the image.
    #cv2.imshow(f'mask {k}', mask)  # Show mask for testing

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # Find contours
    cv2.drawContours(all_contours, cnts, -1, (255, 255, 255), 1)
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

print("number of figures: ", i)

cv2.imwrite('layers/all_contours.png', all_contours)
bn_contours = all_contours[:, :, 0].astype(np.uint8)

kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]], dtype=np.int8)

intersecciones = cv2.filter2D(bn_contours, cv2.CV_8U, kernel)

# Umbralizar para marcar los puntos de intersección
_, intersecciones_bin = cv2.threshold(intersecciones, 250, 255, cv2.THRESH_BINARY)

plt.imshow(intersecciones_bin, cmap='gray')
plt.show()
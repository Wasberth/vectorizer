import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os

# === Step 1: Preprocessing and Edge Sampling ===
def load_image_and_find_contours(image_path):
    og_image = cv2.imread(image_path)
    image = og_image[:,:,0]
    image = 255 - image.reshape((image.shape[0], image.shape[1]))

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours

def sample_contour_points(contour, num_points=500):
    contour = contour[:, 0, :]  # Reshape from (N, 1, 2) to (N, 2)
    # Calculate arc length and interpolate
    sampled = []
    for i in range(num_points):
        #d = (i / (num_points - 1)) * arc_length
        #point = cv2.pointPolygonTest(contour, tuple(contour[0]), True)
        sampled.append(contour[int((i / num_points) * len(contour)) % len(contour)])
    sampled = np.array(sampled)
    return sampled


# === Step 2: Feature Extraction ===
def extract_features(points):
    # Estimate tangent angle and curvature using finite differences
    deltas = np.gradient(points, axis=0)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    second_deltas = np.gradient(deltas, axis=0)
    curvature = np.linalg.norm(second_deltas, axis=1)

    features = np.hstack([points, angles[:, np.newaxis], curvature[:, np.newaxis]])
    return features


# === Step 4: Spline Fitting - I Think ===
def fit_spline(points, degree=3):
    # Parameterize with arc-length
    t = np.linspace(0, 1, len(points))
    x, y = points[:, 0], points[:, 1]

    # Fit spline to the points
    tck, _ = splprep([x, y], s=5.0, k=degree)
    u_fine = np.linspace(0, 1, 200)
    x_fine, y_fine = splev(u_fine, tck)
    return x_fine, y_fine


# === Main pipeline ===
def vectorize_image_to_splines(image_path, output_svg='output.svg'):
    contours = load_image_and_find_contours(image_path)

    plt.switch_backend('Agg')
    plt.figure(figsize=(6, 6))
    for contour in contours:
        sampled_points = sample_contour_points(contour, num_points=100)
        features = extract_features(sampled_points)
        x_spline, y_spline = fit_spline(sampled_points)

        plt.plot(x_spline, y_spline, linewidth=2)

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.clf()
    print(f"Saved SVG to {output_svg}")


# Run the pipeline on a test image
test_image_path = "C:/Users/Wilberth David/Desktop/vectorizer/testimg/3.png"  # Replace with your actual image path
vectorize_image_to_splines(test_image_path, output_svg='C:/Users/Wilberth David/Desktop/vectorizer/testimg/1.svg')
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
import os

# === Step 1: Preprocessing and Edge Sampling ===
def load_image_and_find_contours(image_path):
    og_image = cv2.imread(image_path)
    image = og_image[:, :, 0]
    image = 255 - image.reshape((image.shape[0], image.shape[1]))

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def sample_contour_points(contour, num_points=500):
    contour = contour[:, 0, :]  # Reshape from (N, 1, 2) to (N, 2)
    arc_length = cv2.arcLength(contour, closed=True)
    resampled = []

    for i in range(num_points):
        idx = int((i / num_points) * len(contour))
        resampled.append(contour[idx % len(contour)])
    return np.array(resampled)


# === Step 2: Feature Extraction ===
def extract_features(points):
    deltas = np.gradient(points, axis=0)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    second_deltas = np.gradient(deltas, axis=0)
    curvature = np.linalg.norm(second_deltas, axis=1)

    features = np.hstack([points, angles[:, np.newaxis], curvature[:, np.newaxis]])
    return features


# === Step 3: Clustering Into Curve Segments ===
def cluster_curve_segments(features, eps=5, min_samples=5):
    """
    Cluster features using DBSCAN to identify curve segments.
    Returns a list of index groups corresponding to each cluster.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    unique_labels = set(labels)

    clusters = []
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        indices = np.where(labels == label)[0]
        clusters.append(indices)
    return clusters

def plot_clusters(points, cluster_indices_list, save_path='clusters_debug.png'):
    """
    Plots all input points colored by cluster.
    """
    plt.figure(figsize=(6, 6))
    cmap = cm.get_cmap('tab20')  # Up to 20 distinguishable colors

    for i, cluster_indices in enumerate(cluster_indices_list):
        cluster_points = points[cluster_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=cmap(i % 20), label=f'Cluster {i}', s=10)

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    print(f"Saved cluster debug plot to {save_path}")

# === Step 4: Spline Fitting ===
def fit_spline(points, degree=3):
    if len(points) < degree + 1:
        return points[:, 0], points[:, 1]  # Not enough points for a spline

    x, y = points[:, 0], points[:, 1]
    try:
        tck, _ = splprep([x, y], s=5.0, k=degree)
        u_fine = np.linspace(0, 1, 200)
        x_fine, y_fine = splev(u_fine, tck)
        return x_fine, y_fine
    except:
        # Fallback if spline fails
        return x, y

# === Main pipeline ===
def vectorize_image_to_splines(image_path, output_svg='output.svg'):
    contours = load_image_and_find_contours(image_path)

    plt.switch_backend('Agg')
    plt.figure(figsize=(6, 6))

    i = 0

    for contour in contours:
        sampled_points = sample_contour_points(contour, num_points=200)
        features = extract_features(sampled_points)

        clusters = cluster_curve_segments(features, eps=10, min_samples=5)
        print(len(clusters))

        # Plot clusters
        plot_clusters(sampled_points, clusters, save_path=f'clusters_debug_{i}.png')

        for cluster_indices in clusters:
            cluster_points = sampled_points[cluster_indices]
            if len(cluster_points) < 4:
                continue  # Skip tiny segments

            x_spline, y_spline = fit_spline(cluster_points)
            plt.plot(x_spline, y_spline, linewidth=2)

        i += 1

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.clf()
    print(f"Saved SVG to {output_svg}")


# Run the pipeline on a test image
test_image_path = "C:/Users/Wilberth David/Desktop/vectorizer/testimg/3.png"
output_svg_path = "C:/Users/Wilberth David/Desktop/vectorizer/testimg/1.svg"
vectorize_image_to_splines(test_image_path, output_svg=output_svg_path)

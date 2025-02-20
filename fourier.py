import numpy as np
import cv2

def trace_contour_sampled(image, sample_fraction=0.3):
    """
    Traces a connected sequence of 1s in a binary image using a sampled subset and nearest-neighbor approach.

    Parameters:
        image (2D array): Binary image with values 0 and 1.
        sample_fraction (float): Fraction of points to sample (default 0.3 or 30%).

    Returns:
        list: Sequence of complex numbers representing traced path.
    """
    array = np.array(image)
    rows, cols = array.shape  # Image size

    # Get all nonzero points
    nonzero = np.nonzero(array)
    if len(nonzero[0]) == 0:
        return []  # No 1s found

    # Convert to a list of (row, col) tuples
    points = np.column_stack(nonzero).tolist()

    # Sample a subset of points (30% by default)
    sample_size = max(1, int(len(points) * sample_fraction))
    sampled_points = np.random.choice(len(points), sample_size, replace=False)
    sampled_points = [points[i] for i in sampled_points]

    # Pick an initial point (e.g., first sampled)
    current = sampled_points[0]
    secuence = [complex(*current)]
    sampled_points.remove(current)

    # Iteratively select the closest point
    while sampled_points:
        # Find the closest remaining point
        closest = min(sampled_points, key=lambda p: (p[0] - current[0])**2 + (p[1] - current[1])**2)
        secuence.append(complex(*closest))
        sampled_points.remove(closest)
        current = closest

    return secuence

def trace_contour(image):
    """
    Traces the connected sequence of 1s in a binary image.
    
    Parameters:
        image (2D array): Binary image with values 0 and 1.

    Returns:
        list: Sequence of complex numbers representing traced path.
    """
    array = np.array(image)
    rows, cols = array.shape  # Get shape

    # Find first 1 (optimized)
    nonzero = np.nonzero(array)
    if len(nonzero[0]) == 0:
        return []  # No 1s found

    current = (nonzero[0][0], nonzero[1][0])  # First 1 position
    secuence = [complex(*current)]  # Store as complex number

    # Directions (tuples instead of np.array)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    stack = [current]  # Stack for DFS

    while stack:
        current = stack.pop()
        array[current] = 0  # Mark visited

        for dx, dy in directions:
            nxt = (current[0] + dx, current[1] + dy)

            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and array[nxt] == 1:
                stack.append(nxt)
                secuence.append(complex(*nxt))  # Store as complex number

    return secuence

def fourier(image):
    secuence = trace_contour_sampled(image)

    return np.fft.fft(secuence)

def position(fourier, t):
    sum = 0
    for i in range(len(fourier)):
        sum += fourier[i] * np.exp(-1j * 2 * np.pi * i * t)
    return sum

def velocity(fourier, t):
    sum = 0
    for i in range(len(fourier)):
        sum += fourier[i] * i * np.exp(1j * 2 * np.pi * i * t)
    return 2 * np.pi * 1j * sum

def acceleration(fourier, t):
    sum = 0
    for i in range(len(fourier)):
        sum += fourier[i] * i * i * np.exp(1j * 2 * np.pi * i * t)
    return - 4 * np.pi * np.pi * sum

def jolt(fourier, t):
    sum = 0
    for i in range(len(fourier)):
        sum += fourier[i] * i * i * i * np.exp(1j * 2 * np.pi * i * t)
    return - 8 * np.pi * np.pi * np.pi * 1j * sum

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = cv2.imread('D:/My Files/Documentos/Codigos/ai-image-to-vector/layers/contour0.png', cv2.IMREAD_GRAYSCALE)
    binary = (image == 255).astype(np.uint8)
    fourier_image = fourier(binary)
    time = np.linspace(0, 1, len(fourier_image))

    v_pos = np.array([position(fourier_image, t) for t in time])
    v_vel = np.array([velocity(fourier_image, t) for t in time])
    v_acc = np.array([acceleration(fourier_image, t) for t in time])
    v_jolt = np.array([jolt(fourier_image, t) for t in time])

    plt.plot(v_pos.real, v_pos.imag, label="Position")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.grid()
    plt.legend()
    plt.title("Complex Function Trajectory")
    plt.show()


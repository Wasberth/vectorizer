import numpy as np
import cv2

def trace_contour(image):
    """
    Traces the connected sequence of 1s in a binary image.
    
    Parameters:
        image (2D array): Binary image with values 0 and 1.

    Returns:
        list: Sequence of complex numbers representing traced path.
    """
    array = np.array(image[::-1, :])
    cols, rows = array.shape  # Get shape

    # Find first 1 (optimized)
    nonzero = np.nonzero(array)
    if len(nonzero[0]) == 0:
        return []  # No 1s found

    current = (nonzero[0][0], nonzero[1][0])  # First 1 position
    secuence = [complex(*current[::-1])]  # Store as complex number

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
                secuence.append(complex(*nxt[::-1]))  # Store as complex number

    return secuence

def draw_trace(secuence, t):
    return secuence[int((len(secuence) * t)) % len(secuence)]

def fourier(image):
    secuence = trace_contour(image)

    return np.fft.fftshift(np.fft.fft(secuence))

def position(fourier, t):
    sum = 0
    N = len(fourier)
    for k in range(- N // 2, N // 2):
        sum += fourier[k + N // 2] * np.exp(1j * 2 * np.pi * t * k)
    return sum / N

def velocity(fourier, t):
    sum = 0
    N = len(fourier)
    for k in range(- N // 2, N // 2):
        sum += fourier[k + N // 2] * k * np.exp(1j * 2 * np.pi * k * t)
    return 2 * np.pi * 1j * sum / N

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

    image = cv2.imread('D:/My Files/Documentos/Codigos/ai-image-to-vector/layers/contour6.png', cv2.IMREAD_GRAYSCALE)
    binary = (image == 255).astype(np.uint8)
    #trace = trace_contour(binary)
    fourier_image = fourier(binary)
    time = np.linspace(0, 1, len(fourier_image))

    #ac_pos = np.array([draw_trace(trace, t) for t in time])
    v_pos = np.array([position(fourier_image, t) for t in time])
    v_vel = np.array([velocity(fourier_image, t) for t in time])
    #v_acc = np.array([acceleration(fourier_image, t) for t in time])
    #v_jolt = np.array([jolt(fourier_image, t) for t in time])

    #plt.plot(np.abs(np.fft.fftshift(fourier_image)))
    #plt.plot(np.arctan2(fourier_image.imag, fourier_image.real))

    plt.plot(v_pos.real, v_pos.imag, label="Position")
    plt.plot(v_vel.real, v_vel.imag, label="Velocity")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    #plt.xlim(0, image.shape[1])
    #plt.ylim(0, image.shape[0])
    plt.grid()
    plt.legend()
    plt.title("Complex Function Trajectory")
    plt.show()


import numpy as np
import re
from svgpathtools import parse_path
from svgpathtools import Path

def get_points_on_path_fast(d: str, n: int = 1000):
    """
    Returns a list of n points on the SVG path by sampling evenly in parameter t (fast, not distance-accurate).

    :param d: The 'd' attribute of the SVG path.
    :param n: Number of points to generate (default is 9000).
    :return: A list of (x, y) tuples representing points on the path.
    """
    path: Path = parse_path(d)
    points = []

    for i in range(n):
        # Normalized parameter t from 0 to 1
        t = i / (n - 1) if n > 1 else 0
        point = path.point(t)  # Uses parameter t directly
        points.append(point.real)
        points.append(point.imag)

    return np.array(points)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    file = "D:\My Files\Catalogo\sampling_test.svg"
    with open(file, 'r') as f:
        content = f.read()
        match = re.findall(r'<path([^>]*?)d="([^"]+)"([^>]*?)>', content)
        for m in match:
            if "fil0" in m[0]:
                continue

            points = get_points_on_path_fast(m[1], 500)

            # graph points
            plt.scatter([p[0] for p in points], [-p[1] for p in points])
            plt.show()




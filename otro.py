import numpy as np
import pickle
from utilsCNN import sample_curve, controls_on_line, controls_on_squared

def dtw_distance(X, Y):
    """Calcula la distancia de Dynamic Time Warping entre dos secuencias de puntos X y Y."""
    N = len(X)
    cost = np.zeros((N, N))
    
    # Llenar la matriz de costos con distancias Euclidianas
    cost[0, 0] = np.linalg.norm(X[0] - Y[0])
    
    for i in range(1, N):
        cost[i, 0] = cost[i-1, 0] + np.linalg.norm(X[i] - Y[0])
        cost[0, i] = cost[0, i-1] + np.linalg.norm(X[0] - Y[i])
    
    for i in range(1, N):
        for j in range(1, N):
            cost[i, j] = np.linalg.norm(X[i] - Y[j]) + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    
    return cost[N-1, N-1]

def circular_dtw_distance(A, B):
    """
    Calcula la distancia Circular Dynamic Time Warping (cDTW) entre dos figuras cerradas A y B
    representadas como secuencias de puntos.
    
    A y B deben ser arrays de forma (N, 2) donde N es el número total de puntos muestreados.
    """
    N = len(A)
    
    # Verificar que ambas secuencias tengan el mismo número de puntos
    assert len(A) == len(B), "Las secuencias de puntos deben tener el mismo número de puntos."
    
    min_dtw = np.inf
    
    # Rotar B y calcular DTW con A
    for k in range(N):
        rotated_B = np.roll(B, shift=k, axis=0)
        dtw = dtw_distance(A, rotated_B)
        min_dtw = min(min_dtw, dtw)
    
    return min_dtw

def sample_closed_bezier_figure_fixed_total(points, total_samples):
    """
    Muestrea una figura cerrada representada por una secuencia de curvas Bézier cúbicas con
    un número total de muestras fijo.
    
    points: np.array de forma (N, 2), donde N % 3 == 0
    total_samples: número total de puntos a muestrear en toda la figura
    """
    real_number_of_points = 0
    reshaping = False
    for i in range(points.shape[0]):
        if points[i][0] == -1 and points[i][1] == -1:
            reshaping = True
            break
        real_number_of_points += 1
    
    if reshaping:
        points = points[:real_number_of_points]
        if real_number_of_points % 3 == 1:
            add = np.zeros((1, 2))
            c1, c2 = controls_on_squared(points[-2], points[-1], points[0])
        if real_number_of_points % 3 == 2:
            add = np.zeros((2, 2))
            add[0] = points[-1]
            add[1] = points[0]
            c1, c2 = controls_on_squared(add)
        points = np.concatenate((points, add))
        points[-2] = c1
        points[-1] = c2
    
    # num_curves = points.shape[0] // 3
    # 
    # # Distribuir los samples por curva (redondeando)
    # base_samples = total_samples // num_curves
    # extras = total_samples % num_curves  # algunos tramos tendrán 1 punto más
    # 
    # sampled_points = []
    # 
    # for i in range(num_curves):
    #     curve_points = np.zeros((4,2))
    #     curve_points[0][0] = points[3*i]
    #     curve_points[0][1] = points[3*i + 1]
    #     curve_points[1][0] = points[3*i + 2]
    #     curve_points[1][1] = points[0] if i == num_curves - 1 else points[3*i + 3]
    # 
    #     # Asignar la cantidad de puntos a esta curva
    #     n_samples = base_samples + (1 if i < extras else 0)
    # 
    #     # Muestrear y omitir el último punto para evitar duplicados entre curvas
    #     curve = sample_curve(curve_points, n_points=n_samples)
    #     if i < num_curves - 1:
    #         curve = curve[:-1]  # evitar duplicar el primer punto del siguiente tramo
    #     sampled_points.append(curve)
    # 
    # return np.vstack(sampled_points)


root = 'dataset/'
tests = [0, 1]
with open(root+'base_file_shapes.pkl', 'rb') as file:
    shape_dict = pickle.load(file)
output_shape = shape_dict['output']
output_matrix = np.memmap(root+'outputTransformer/0.npy', dtype=np.float64, mode='r', shape=output_shape)
test_1 = np.array(output_matrix[tests[0]])
test_2 = np.array(output_matrix[tests[1]])
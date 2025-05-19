import numpy as np
import matplotlib.pyplot as plt

from FourierContourNN import preprocess_fft, inverse_preprocess_fft

file_input = 'D:/My Files/Documentos/Codigos/ai-image-to-vector/dataset/input_samples/Adventure Time_fig_6.npy'
points = np.load(file_input)

reconstructed = inverse_preprocess_fft(preprocess_fft(points))

_, (ax1, ax2) = plt.subplots(1, 2)

points = points.reshape((-1, 2))
reconstructed = reconstructed.reshape((-1, 2))

ax1.plot(points[:, 0], points[:, 1], 'o')
ax1.plot(points[:, 0][0], points[:, 1][0], marker='o', color='r')
ax1.set_title('Points from input')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')
ax1.grid(True)

ax2.plot(reconstructed[:, 0], reconstructed[:, 1], 'o')
ax2.plot(reconstructed[:, 0][0], reconstructed[:, 1][0], marker='o',  color='r')
ax2.set_title('Points reconstructed')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.grid(True) 

plt.show()
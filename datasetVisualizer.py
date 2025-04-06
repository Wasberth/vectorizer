import numpy as np
import matplotlib.pyplot as plt
import utilsFCN

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

dataset_sufix = 0
test_id = 1
file = utilsFCN.input_directory+f'{dataset_sufix}.npy'
input_shape = utilsFCN.getInputShape(file)
input = np.memmap(file, mode='r', shape=input_shape)
output_shape = (input_shape[0], input_shape[1], input_shape[2], 3)
file = utilsFCN.output_directory+f'{dataset_sufix}.npy'
output = np.memmap(file, mode='r', shape=output_shape)

test = np.transpose(input[test_id], (2, 0, 1))
colors = [(0,0,0), (255, 255, 255)]

ax1.set_title("Imagen original")
ax2.set_title("Pixeles SE")
ax3.set_title("Pixeles Control")
ax4.set_title("Pixeles basura")
ax1.imshow(utilsFCN.drawImageFromArray(test[0], colors))
test = np.transpose(output[test_id], (2, 0, 1))
colors = [(0,0,0), (255, 255, 255)]
ax2.imshow(utilsFCN.drawImageFromArray(test[0], colors))
ax3.imshow(utilsFCN.drawImageFromArray(test[1], colors))
ax4.imshow(utilsFCN.drawImageFromArray(test[2], colors))
plt.show()
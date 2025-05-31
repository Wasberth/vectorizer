import os
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = 'ContNNv5'
history = {
    'mae': np.load(f'history/{MODEL_NAME}_mae.npy'),
    'mse': np.load(f'history/{MODEL_NAME}_mse.npy'),
    'val_mae': np.load(f'history/{MODEL_NAME}_val_mae.npy'),
    'val_mse': np.load(f'history/{MODEL_NAME}_val_mse.npy')
}

print("MAE:", history['mae'][-1])
print("MSE:", history['mse'][-1])
print("VAL MAE:", history['val_mae'][-1])
print("VAL MSE:", history['val_mse'][-1])

for key in history:
    history[key] = np.log(history[key])

# Clear plots
plt.clf()

# summarize history for accuracy
plt.plot(history['mae'])
plt.plot(history['val_mae'])
plt.title('Mean Absolute Error')
plt.ylabel('mae (log)')
plt.xlabel('época')
plt.legend(['entrenamiento', 'validación'], loc='upper left')
plt.savefig(f'plots/temp1.png')

# clear plot
plt.clf()

# summarize history for loss
plt.plot(history['mse'])
plt.plot(history['val_mse'])
plt.title('Mean Squared Error')
plt.ylabel('mse (log)')
plt.xlabel('época')
plt.legend(['entrenamiento', 'validación'], loc='upper left')
plt.savefig(f'plots/temp2.png')
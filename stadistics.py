import pickle
import matplotlib.pyplot as plt

LOSS_DICT = {
    'CNN': 'Mean Squared Error',
    'Autoencoder_2': 'Binary Focal Crossentropy',
    'Autoencoder_5': 'Binary Focal Crossentropy',
    'U-Net': 'Binary Crossentropy',
    'FCN_Focal': 'Binary Focal Crossentropy',
    'FCN_No_Focal': 'Binary Crossentropy'
}

METRICS_DICT = {
    'accuracy': 'Accuracy',
    'io_u': 'Intersection over union',
    'recall': 'Recall',
    'mean_absolute_error': 'Mean Absolute Error',
    'mean_square_error': 'Mean Squared Error'
}

model = 'FCN_Focal'
model_name = 'Fully Convolutional Network (Binary Focal Crossentropy)'
performance_dict = {}

if model == 'FCN_Focal':
    with open('models/performance_'+model+'_1.pkl', 'rb') as file:
        performance_dict = pickle.load(file)
    with open('models/performance_'+model+'_2.pkl', 'rb') as file:
        temp = pickle.load(file)
    
    for key in performance_dict.keys():
        performance_dict[key] = performance_dict[key]+ temp[key]
else:
    with open('models/performance_'+model+'.pkl', 'rb') as file:
        performance_dict = pickle.load(file)

for key, value in performance_dict.items():
    if key.startswith('val_'):
        continue
    ylabel = ''
    if key == 'loss':
        ylabel = LOSS_DICT[model]
    else:
        ylabel = METRICS_DICT[key]
    plt.plot(value)
    plt.plot(performance_dict['val_'+key])
    plt.title(model_name)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()
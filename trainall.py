from FourierContourNN import main as trainPreprocessContourNN
from FourierContourNN import preprocess_fft, normalize

EPOCHS = 100000
BATCH_SIZE = 64
RECORD = 'stats.txt'

stat = trainPreprocessContourNN(
    'ContNNv5', 75, BATCH_SIZE, None, 75
)
with open(RECORD, 'a') as f:
    f.write(f"Best model for ContNNv5 was saved at epoch: {stat[0]}. Best mse: {stat[1]}. Best mae: {stat[2]}. Best val_mse: {stat[3]}. Best val_mae: {stat[4]}\n")

stat = trainPreprocessContourNN(
    'FourierContNNv3', 75, BATCH_SIZE, preprocess_fft, 75
)
with open(RECORD, 'a') as f:
    f.write(f"Best model for FourierContNNv3 was saved at epoch: {stat[0]}. Best mse: {stat[1]}. Best mae: {stat[2]}. Best val_mse: {stat[3]}. Best val_mae: {stat[4]}\n")

stat = trainPreprocessContourNN(
    'NormalContNNv2', 160, BATCH_SIZE, normalize, 160
)
with open(RECORD, 'a') as f:
    f.write(f"Best model for NormalContNNv2 was saved at epoch: {stat[0]}. Best mse: {stat[1]}. Best mae: {stat[2]}. Best val_mse: {stat[3]}. Best val_mae: {stat[4]}\n")
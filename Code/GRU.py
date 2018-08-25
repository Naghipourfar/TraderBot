import h5py
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.layers import GRU, Input
from keras.models import Model

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

with h5py.File(''.join(['bitcoin2015to2017_close.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value

output_file_name = 'bitcoin2015to2017_close_GRU_1_tanh_relu_'

step_size = datas.shape[1]
units = 150
batch_size = 8
n_features = datas.shape[2]
epochs = 250
output_size = 12
training_size = datas.shape[0] - 1
# training_size = int(0.8 * datas.shape[0])
training_datas = datas[:training_size, :]
training_labels = labels[:training_size, :, :]
validation_datas = datas[training_size:, :]
validation_labels = labels[training_size:, :, :]
print(training_labels.shape)

validation_size = datas.shape[0] - training_size

training_labels = [np.array(training_labels[:, :, 0]).reshape((training_size, -1)),
                   np.array(training_labels[:, :, 1]).reshape((training_size, -1)),
                   np.array(training_labels[:, :, 2]).reshape((training_size, -1))]

validation_labels = [np.array(validation_labels[:, :, 0]).reshape((validation_size, -1)),
                     np.array(validation_labels[:, :, 1]).reshape((validation_size, -1)),
                     np.array(validation_labels[:, :, 2]).reshape((validation_size, -1))]

print("Training Shape\t:\t(%d, %d, %d)" % (training_size, step_size, n_features))
print("Validation Shape\t:\t(%d, %d, %d)" % (datas.shape[0] - training_size, step_size, n_features))

input_layer = Input(shape=(step_size, n_features,))
layer_1 = GRU(units=units, return_sequences=True)(input_layer)
layer_1 = Dropout(0.5)(layer_1)

layer_2 = GRU(units=units, return_sequences=False)(layer_1)
layer_2 = Dropout(0.5)(layer_2)

output_1 = Dense(output_size, activation="tanh", name="close_dense")(layer_2)
output_2 = Dense(output_size, activation="tanh", name="high_dense")(layer_2)
output_3 = Dense(output_size, activation="tanh", name="low_dense")(layer_2)

model = Model(inputs=input_layer, outputs=[output_1, output_2, output_3])
model.compile(optimizer="adam", loss=["mse", "mse", "mse"], loss_weights=[0.001, 0.001, 0.001])
model.fit(training_datas,
          training_labels,
          batch_size=batch_size,
          validation_data=(validation_datas, validation_labels),
          epochs=epochs,
          verbose=2,
          callbacks=[
              CSVLogger(output_file_name + '.csv', append=True),
              ModelCheckpoint(output_file_name + '-{epoch:02d}-{val_loss:.8f}.hdf5', monitor='val_loss',
                              verbose=1, mode='min', save_best_only=True)])

from shutil import copy2
import numpy as np

np.random.seed(123)

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten, Concatenate
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from prednet import PredNet
from data_utils import SequenceGenerator
from settings import *


# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)


orig_weights = os.path.join(MODELS_DIR, 'prednet_weights.hdf5')  # original t+1 weights
orig_model = os.path.join(MODELS_DIR, 'prednet_model.json') # original model
extrap_weights = os.path.join(MODELS_DIR, 'prednet_weights-extrapfinetuned.hdf5')  # Where the new weights will be saved.
extrap_model = os.path.join(MODELS_DIR, 'prednet_model-extrapfinetuned.json') # Where the new model will be saved.

# Data file directories
train_file = os.path.join(DATA_DIR, 'train_data.hkl')
train_sources = os.path.join(DATA_DIR, 'train_sources.hkl')
val_file = os.path.join(DATA_DIR, 'val_data.hkl')
val_sources = os.path.join(DATA_DIR, 'val_sources.hkl')
env_val_data = None
env_train_data = None
if ENV_DATA:
    env_val_data = os.path.join(DATA_DIR, 'env_val_data.hkl')
    env_train_data = os.path.join(DATA_DIR, 'env_train_data.hkl')

# Training parameters
nt = NT
extrap_start_time = EXTRAP  # starting at this time step, the prediction from the previous time step will be treated as the actual input
nb_epoch = NB_EPOCH
batch_size = BATCH_SIZE
samples_per_epoch = SAMPLES_PER_EPOCH
N_seq_val = N_SEQ_VAL
lr = LR

# Load the original t+1 model
f = open(orig_model, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
orig_model.load_weights(orig_weights)

layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(input_shape)
predictions = prednet(inputs)

if ENV_DATA:
    inputs_env = Input(shape=(nt,1,))
    flatten_inputs = Flatten()(inputs_env)
    another_dense = Dense(1, trainable=False)(flatten_inputs)

    # Concatenating the two inputs
    concat_layer = Concatenate()([another_dense, predictions])

    # Define final layer and create and compile the model
    output_dense = Dense(1, trainable=False)(concat_layer)
    model = Model(inputs=[inputs, inputs_env], outputs=output_dense)
else:
    model = Model(inputs=inputs, outputs=predictions)

model.compile(loss=extrap_loss, optimizer='adam')

# Create the Sequence Generators, found in data_utils. These create the batches used while training the model.
train_generator = SequenceGenerator(train_file, train_sources, env_train_data, nt, batch_size=batch_size, shuffle=True, output_mode='prediction')
val_generator = SequenceGenerator(val_file, val_sources, env_val_data, nt, batch_size=batch_size, N_seq=N_seq_val, output_mode='prediction')

lr_schedule = lambda \
    epoch: lr if epoch < nb_epoch / 2 else 0.1 * lr  # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if not os.path.exists(MODELS_DIR): os.mkdir(MODELS_DIR)
callbacks.append(ModelCheckpoint(filepath=extrap_weights, monitor='val_loss', save_best_only=True))
model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks, validation_data=val_generator, validation_steps=N_seq_val / batch_size)

# Save the newly trained model
json_string = model.to_json()
with open(extrap_model, "w") as f:
    f.write(json_string)
output_dir = "Extrap_model_nt" + str(nt) + "_b" + str(batch_size) + "_e" + str(nb_epoch) + "_s" + str(samples_per_epoch) + "_v" + str(N_seq_val) + "_lr" + str(lr)
output_dir = os.path.join(MODELS_DIR, output_dir)
if not os.path.exists(output_dir): os.mkdir(output_dir)
copy2(extrap_weights, output_dir)
copy2(extrap_model, output_dir)

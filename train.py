from shutil import copy2
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Concatenate
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from prednet import PredNet
from data_utils import SequenceGenerator
from settings import *
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# The directories where the resulting trained model and weights are saved.
save_weights = os.path.join(MODELS_DIR, 'prednet_weights.hdf5')
save_model = os.path.join(MODELS_DIR, 'prednet_model.json')

# Loads the files training and validation files.
train_file = os.path.join(DATA_DIR, 'train_data.hkl')
train_sources = os.path.join(DATA_DIR, 'train_sources.hkl')
val_file = os.path.join(DATA_DIR, 'val_data.hkl')
val_sources = os.path.join(DATA_DIR, 'val_sources.hkl')
env_val_data = None
env_train_data = None
if ENV_DATA:
    env_val_data = os.path.join(DATA_DIR, 'env_val_data.hkl')
    env_train_data = os.path.join(DATA_DIR, 'env_train_data.hkl')

# Hyper parameters. Original values included beside each. Global variables are contained in settings.py
nb_epoch = NB_EPOCH # orig:100
batch_size = BATCH_SIZE # orig: 4
samples_per_epoch = SAMPLES_PER_EPOCH # orig: 125
N_seq_val = N_SEQ_VAL # orig: 100; number of sequences to use for validation
nt = NT # orig: 20; number of time steps used in image sequences
lr = LR # orig: 0.02

# Image parameters, global variable are contained in settings.py
image_height = HEIGHT
image_width = WIDTH
n_channels = N_CHANNELS

if K.image_data_format() == 'channels_first':
    input_shape = (n_channels, image_height, image_width)
else:
    input_shape = (image_height, image_width, n_channels)

# Model parameters for PredNet
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3) 		# orig: (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3) 	# orig: (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3) 	# orig: (3, 3, 3, 3)
extrap = None # Not used in training.

# This is the weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.array([1., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)

# All time steps are to be weighted equally, except the first
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0

# Start with lr of 0.002 and then drop to 0.0002 after 50 epochs
lr_schedule = lambda epoch: lr if epoch < nb_epoch/2 else 0.1*lr
callbacks = [LearningRateScheduler(lr_schedule)]
if not os.path.exists(MODELS_DIR): os.mkdir(MODELS_DIR)
callbacks.append(ModelCheckpoint(filepath=save_weights, monitor='val_loss', save_best_only=True)) # orig: save_best_only=True

# Setting up the layers of the model used for the satellite data as input. Each layer takes the previous layer as an input. Uses Keras Functional API
inputs = Input(shape=(nt,) + input_shape)
prednet = PredNet(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, output_mode='error', extrap_start_time=extrap, return_sequences=True)
prednet_layer = prednet(inputs)  # Shape will be (batch_size, nt, nb_layers)
dense_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(prednet_layer)  # Calculates the weighted error by layer
flatten = Flatten()(dense_time)  # Shape will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(flatten)  # Calculates the weighted errors by time

# Setting up the layers of the model used for the numerical data as input. This step is optional.
if ENV_DATA:
    inputs_vector = Input(shape=(nt,1,))
    flatten_inputs = Flatten()(inputs_vector)
    another_dense = Dense(1, trainable=False)(flatten_inputs)

    # Concatenating the two inputs
    concat_layer = Concatenate()([another_dense, final_errors])

    # Define final layer and create and compile the model
    output_dense = Dense(1, trainable=False)(concat_layer)
    model = Model(inputs=[inputs, inputs_vector], outputs=output_dense)
else:
    model = Model(inputs=inputs, outputs=final_errors)

model.compile(loss='mse', optimizer='adam') # orig: adam
# Create the Sequence Generators, found in data_utils. These create the batches used while training the model
train_generator = SequenceGenerator(train_file, train_sources, env_train_data, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, env_val_data, nt, batch_size=batch_size, N_seq=N_seq_val)
# This step trains the model.
model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks, validation_data=val_generator, validation_steps=N_seq_val / batch_size)

# Save the model to the directory specified in save_model
if not os.path.exists(save_model):
    json_string = model.to_json()
    with open(save_model, "w") as f:
        f.write(json_string)

    # Name of saved model, based on hyper parameters used
    outputdir = "nt"+str(nt)+"_batch"+str(batch_size)+"_epoch"+str(nb_epoch)+"_samples"+str(samples_per_epoch)+"_valSeq"+str(N_seq_val)+"_lr"+str(lr)
    outputdir = os.path.join(MODELS_DIR, outputdir)

    if not os.path.exists(outputdir): os.mkdir(outputdir)
    copy2(save_weights, outputdir)
    copy2(save_model, outputdir)

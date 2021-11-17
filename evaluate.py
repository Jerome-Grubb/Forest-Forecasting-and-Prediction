import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Model, model_from_json
from keras.layers import Input
from skimage.measure import compare_ssim
from prednet import PredNet
from data_utils import SequenceGenerator
from scipy.misc import imsave
from settings import *

# Hyper parameters
n_plot = 10
batch_size = BATCH_SIZE
nt = NT
numtests = 18
extrap = None

# Argument that dictates how many steps the prediction is if a finetuned model is being evaluated.
parser = argparse.ArgumentParser()
parser.add_argument('-ft', help="fine-tune multistep: add extrap time")
args = parser.parse_args()

# Load model and testing data and files
weights_file = os.path.join(MODELS_DIR, 'prednet_weights.hdf5')
trained_model = os.path.join(MODELS_DIR, 'prednet_model.json')
test_file = os.path.join(DATA_DIR, 'test_data.hkl')
test_sources = os.path.join(DATA_DIR, 'test_sources.hkl')
env_test_data = None
if ENV_DATA:
    env_test_data = os.path.join(DATA_DIR, 'env_test_data.hkl')

# If a finetuned model is being used.
if args.ft is not None:
    extrap = int(args.ft)
    nt = extrap + 5
    weights_file = os.path.join(MODELS_DIR, 'prednet_weights-extrapfinetuned.hdf5')
    trained_model = os.path.join(MODELS_DIR, 'prednet_model-extrapfinetuned.json')

# Load the trained model
f = open(trained_model, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])

# Setting up the layers of the model
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
inputs_vector = Input(shape=(nt, 1,))
if ENV_DATA:
    test_model = Model(inputs=[inputs, inputs_vector], outputs=predictions)
    test_generator = SequenceGenerator(test_file, test_sources, env_test_data, nt, sequence_start_mode='all',
                                       data_format=data_format)  # orig: unique
else:
    test_model = Model(inputs=inputs, outputs=predictions)
    test_generator = SequenceGenerator(test_file, test_sources, env_test_data, nt, sequence_start_mode='all',
                                       data_format=data_format)  # orig: unique

# Make the predictions
X_test = test_generator.create_all()
X_predictions = test_model.predict(X_test, batch_size)

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
if ENV_DATA:
    mse_model = np.mean((X_test[0][:, 1:] - X_predictions[:, 1:]) ** 2)  # look at all timesteps except the first.
    mse_prev = np.mean((X_test[0][:, :-1] - X_test[0][:, 1:]) ** 2)
    ssim, _ = compare_ssim(X_test[0][:, 1:], X_predictions[:, 1:], full=True,
                           multichannel=True)  # make sure to include all of the metrics.
else:
    mse_model = np.mean((X_test[:, 1:] - X_predictions[:, 1:]) ** 2)
    mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)
    ssim, _ = compare_ssim(X_test[:, 1:], X_predictions[:, 1:], full=True,
                           multichannel=True)

# Save evaluation metrics to a file.
if not os.path.exists(RESULTS_DIR): os.mkdir(RESULTS_DIR)
f = open(os.path.join(RESULTS_DIR, 'prediction_scores.txt'), 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f\n" % mse_prev)
f.write("Model SSIM: %f\n" % ssim)
f.close()

# Plot the predictions
aspect_ratio = float(X_predictions.shape[2]) / X_predictions.shape[3]
plt.figure(figsize=(nt, 2 * aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        if ENV_DATA:
            plt.imshow(X_test[0][i, t], interpolation='none')
        else:
            plt.imshow(X_test[i, t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
                        labelleft='off')
        if t == 0: plt.ylabel('Actual', fontsize=10)
        plt.subplot(gs[t + nt])
        plt.imshow(X_predictions[i, t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
                        labelleft='off')
        if t == 0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir + 'plot_' + str(i) + '.jpg')
    plt.clf()

# Save the plotted predictions
for test in range(numtests):
    testdir = "tile-" + str(test)
    testdir = os.path.join(plot_save_dir, testdir)
    if not os.path.exists(testdir): os.mkdir(testdir)
    for t in range(nt):
        imsave(testdir + "/pred-%02d.jpg" % (t,), X_predictions[test, t])
        if ENV_DATA:
            imsave(testdir + "/orig-%02d.jpg" % (t,), X_test[0][test, t])
        else:
            imsave(testdir + "/orig-%02d.jpg" % (t,), X_test[test, t])

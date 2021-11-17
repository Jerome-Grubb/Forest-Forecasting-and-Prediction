import os

# These are the dimensions of the image.
WIDTH=64
HEIGHT=64
N_CHANNELS=3

# Hyperparameters
NT=10
EXTRAP=5
NB_EPOCH=100
SAMPLES_PER_EPOCH=125
N_SEQ_VAL=32
BATCH_SIZE=4
LR=0.002
ENV_DATA = False

# These are for the evaluation stage.
NUM_TESTS=10
NUM_PLOTS=10

# These are the locations of important directories.
current = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(current, 'Data')
MODELS_DIR = os.path.join(current, 'Models')
RESULTS_DIR = os.path.join(current, 'Exports')

# These are where the results of the prepossessing stage are saved
TRAIN_DIR = os.path.join( DATA_DIR, 'Train')
VAL_DIR = os.path.join( DATA_DIR, 'Val')
TEST_DIR = os.path.join( DATA_DIR, 'Test')

# Used by the scrapper tool to obtain Landsat data
COORDS = []
ZOOM = 12.2
KEY = 'aripuana'

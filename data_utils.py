import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
from random import randint
from scipy.ndimage import imread
from settings import *

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, env_data, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        self.X = hkl.load(data_file)  # X will be a list of directories
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        if ENV_DATA:
            self.env_data = hkl.load(env_data)
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        self.im_shape = (64, 64, 3)

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.im_shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.im_shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        batch_int = np.zeros((current_batch_size, self.nt) + (1,), np.float32) # batch for integer inputs
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            sample = np.zeros((len(self.X[idx:idx + self.nt]),) + self.im_shape)
            count = 0

            for j in self.X[idx:idx + self.nt]:
                tile = imread(j, mode="RGB")
                sample[count] = self.preprocess(tile)
                count += 1
            batch_x[i] = sample

            if ENV_DATA:
                sample_int = np.zeros((len(self.X[idx:idx + self.nt]),) + (1,))
                for j in self.env_data[idx:idx + self.nt]:
                    sample_int[count] = j
                    count += 1
                batch_int[i] = sample_int

        if ENV_DATA:
            batch_x = [batch_x, batch_int]
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        batch_ints = np.zeros((self.N_sequences, self.nt) + (1,), np.float32) # batch for integer inputs
        for i, idx in enumerate(self.possible_starts):
            sample = np.zeros((len(self.X[idx:idx + self.nt]),) + self.im_shape)
            count = 0

            for j in self.X[idx:idx + self.nt]:
                tile = imread(j, mode="RGB")
                sample[count] = self.preprocess(tile)
                count += 1

            if ENV_DATA:
                sample_int = np.zeros((len(self.X[idx:idx + self.nt]),) + (1,))
                for j in self.env_data[idx:idx + self.nt]:
                    sample_int[count] = j
                    count += 1
                batch_ints[i] = sample_int

            X_all[i] = sample
        if ENV_DATA:
            X_all = [X_all, batch_ints]
        return X_all

import numpy as np
from scipy.misc import imresize, imsave
from scipy.ndimage import imread
import hickle as hkl
from settings import *
import csv

image_size = (HEIGHT, WIDTH)
categories = ['all']

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# Adds the file locations for all of the validation data into a list
val_images = []
try:
    _, directory, _ = os.walk(VAL_DIR).next()
    for i in directory:
        val_images.append(('all', os.path.join(VAL_DIR, i)))
except StopIteration:
    pass


# Add the file locations for all of the testing data into a list
test_images = []
try:
    _, directory, _ = os.walk(TEST_DIR).next()
    for i in directory:
        test_images.append(('all', os.path.join(TEST_DIR, i)))
except StopIteration:
    pass

train_images = []
try:
    _, directory, _ = os.walk(TRAIN_DIR).next()
    for i in directory:
        train_images.append(('all', os.path.join(TRAIN_DIR, i)))
except StopIteration:
    pass

# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_images
    splits['test'] = test_images
    splits['train'] = train_images
    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(folder + '/')
            try:
                _, _, files = os.walk(im_dir).next()
                im_list += [im_dir + f for f in sorted(files)]
                source_list += [category + '-' + folder] * len(files)
            except StopIteration:
                pass
        print 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images'
        # X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        X = []
        for _, im_file in enumerate(im_list):
            try:
                im = imread(im_file, mode='RGB')
                X.append(im_file)
                # Process the image and then save it.
                new_im = process_im(im)
                imsave(im_file, new_im)
            except IOError:
                pass
        hkl.dump(X, os.path.join(DATA_DIR, split + '_data.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, split + '_sources.hkl'))

    if ENV_DATA:
        test_data = []
        train_data = []
        val_data = []
        with open('test_data.csv') as inputfile:
            for row in csv.reader(inputfile):
                test_data.append(row[0])
        with open('train_data.csv') as inputfile:
            for row in csv.reader(inputfile):
                train_data.append(row[0])
        with open('val_data.csv') as inputfile:
            for row in csv.reader(inputfile):
                val_data.append(row[0])
        test_data[0] = test_data[0].replace("\xef\xbb\xbf", "")
        test_data = map(int, test_data)
        train_data[0] = train_data[0].replace("\xef\xbb\xbf", "")
        train_data = map(int, train_data)
        val_data[0] = val_data[0].replace("\xef\xbb\xbf", "")
        val_data = map(int, val_data)
        hkl.dump(test_data, os.path.join(DATA_DIR, 'env_test_data.hkl'))
        hkl.dump(train_data, os.path.join(DATA_DIR, 'env_train_data.hkl'))
        hkl.dump(val_data, os.path.join(DATA_DIR, 'env_val_data.hkl'))

# resize and crop the image
def process_im(im):
    target_ds = float(image_size[0])/im.shape[0]
    im = imresize(im, (image_size[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - image_size[1]) / 2)
    im = im[:, d:d+image_size[1]]
    return im


if __name__ == '__main__':
    process_data()

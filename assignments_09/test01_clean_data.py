from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

root = r"D:\python\training\NLP\assignments_nlp\assignments_09"
train_folders = [os.path.join(root, "notMNIST_large"+"\\"+i) for i in filter(lambda x: "pickle" not in x,os.listdir(os.path.join(root,"notMNIST_large")))]
test_folders = [os.path.join(root, "notMNIST_small"+"\\"+i) for i in filter(lambda x: "pickle" not in x,os.listdir(os.path.join(root,"notMNIST_small")))]

for folers in [train_folders, test_folders]:
    plt.figure(figsize=(10, 8))
    for row, dir_i in enumerate(folers):
        images = os.listdir(dir_i)
        for col, i in enumerate(images[:10]):
            plt.subplot(10, 10, row*10+col+1)
            img = imread(os.path.join(dir_i, i))
            plt.xticks([])
            plt.yticks([])
            imshow(img, cmap='Greys_r')
    plt.show()

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickl.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_folders = [os.path.join(root, "notMNIST_large" + "\\" + i) for i in
                 filter(lambda x: "pickle" not in x, os.listdir(os.path.join(root, "notMNIST_large")))]
test_folders = [os.path.join(root, "notMNIST_small" + "\\" + i) for i in
                filter(lambda x: "pickle" not in x, os.listdir(os.path.join(root, "notMNIST_small")))]

train_datasets = maybe_pickle(train_folders, 20000)
test_datasets = maybe_pickle(test_folders, 1000)

for folder in [train_datasets,test_datasets]:
    plt.figure(figsize = (10,8))
    for row,value in enumerate(folder):
        with open(value,'rb') as f:
            dataset = pickle.load(f)
        for i,j in enumerate(range(1000,1000+10)):
            plt.subplot(10,10,row*10+i+1)
            img = dataset[j,:,:]
            plt.xticks([])
            plt.yticks([])
            imshow(img,cmap='Greys_r')
    plt.show()


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)

_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


for data in [[train_dataset,train_labels],[test_dataset, test_labels],[valid_dataset, valid_labels],]:
    imgs, labels = data
    plt.figure(figsize = (10,8))
    for row,value in enumerate(imgs[:100]):
        plt.subplot(10,10,row+1)
        plt.title(labels[row])
        plt.xticks([])
        plt.yticks([])
        imshow(value,cmap='Greys_r')
    plt.show()

data_root = r'D:\python\training\NLP\assignments_nlp\assignments_09'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
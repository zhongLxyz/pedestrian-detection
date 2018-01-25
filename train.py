
"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle, build_hdf5_image_dataset, image_preloader
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import h5py

# Load the data set
# dataset_file = 'my_dataset.txt'
# testset_file = 'my_testset.txt'

# Extract images from dataset using h5py
h5f = h5py.File('dataset.h5', 'r')
X = h5f.get('X')
Y = h5f.get('Y')

testh5py = h5py.File('testset.h5', 'r')
X_test = testh5py.get('X')
Y_test = testh5py.get('Y')


# Shuffle the data
X, Y = shuffle(X, Y)

# Normalize the images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Rotate/flip/blur the images in dataset in increase 'noise'
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)


# Define our network architecture:
# Input is a 32x32 image with 3 color channels (RGB)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: 512 node in hidden layer
network = fully_connected(network, 512, activation='relu')

# Step 7: Randomly dropout values to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Finally 2 output nodes for class..
# 0 = not a pedestrian
# 1 = is a pedestrian
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0,
                    checkpoint_path='./checkpoint/ped-classifier.tfl.ckpt')

# Train the neural network (for 30 epochs)
# X is the images, Y is the labels
# X_test and Y_test are the same but for validation
model.fit(X, Y, n_epoch=30, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='ped-classifier')

# Save model when training is complete to a file
model.save("./model/ped-classifier.tfl")
print("Network trained and saved as ped-classifier.tfl!")

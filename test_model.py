
from __future__ import division, print_function, absolute_import
# Import necessary libraries
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import h5py

# Extract images from testset.h5
testh5py = h5py.File('testset.h5', 'r')
X_test = testh5py.get('X')
Y_test = testh5py.get('Y')

# Shuffle testing data
X_test, Y_test = shuffle(X_test, Y_test)


# Same network definition as train.py
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoint/ped-classifier.tfl.ckpt')

# Load previously trained model
model.load("./model/ped-classifier.tfl")

# Test the trained networking using samples from testset.h5
num_correct = 0
num_total = 0
num_pedestrian = 0
num_non_pedestrian = 0
num_pedestrian_correct = 0
num_non_pedestrian_correct = 0
for i in range(len(X_test)):
    # Predict
    prediction = model.predict([X_test[i]])

    # Check the result.
    # is_pedestrian = (np.argmax(prediction[0]) == np.argmax(Y_test[i]))

    # NOTE: prediction is the actual output, Y_test[i] is the desired output
    # Both prediction and Y_test[i] are np.ndarray objects

    # Class of desired output
    actual_class = np.argmax(Y_test[i])

    # Check class of actual output
    predicted_class = 0
    if np.argmax(prediction[0]) == 1:
        predicted_class = 1

    # Check if actual output is correct/wrong
    if actual_class == 0:
        num_non_pedestrian += 1
        if predicted_class == 0:
            num_non_pedestrian_correct += 1

    if actual_class == 1:
        num_pedestrian += 1
        if predicted_class == 1:
            num_pedestrian_correct += 1


# Print confusion matrix
print("Confusion Matrix: \n" + \
    "Non pedestrians images: [" + str(num_non_pedestrian_correct) +
    " classified as non pedestrian] / [" +
    str(num_non_pedestrian - num_non_pedestrian_correct) +" classified as pedestrian]\n" +
    "Pedestrians images: [" + str(num_pedestrian - num_pedestrian_correct) + " classified as non pedestrian] / [" + str(num_pedestrian_correct) +" classified as pedestrian]\n")

# Print accuracy of network
num_correct = num_pedestrian_correct + num_non_pedestrian_correct
num_total = num_pedestrian + num_non_pedestrian
print("Accuracy: " + str(num_correct/num_total))

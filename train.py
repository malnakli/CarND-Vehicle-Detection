import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from utils import extract_features
from parameters import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle

# Read in cars and notcars
cars = glob.glob('data/vehicles/*/*.png')
notcars = glob.glob('data/non-vehicles/*/*.png')


def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    example_img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


data_info = data_look(cars, notcars)

print(data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"],
      ' and data type:', data_info["data_type"])


car_features = extract_features(cars, color_space=COLOR_SPACE,
                                spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                                orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                                cell_per_block=CELL_PER_BLOCK,
                                hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                                hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
notcar_features = extract_features(notcars, color_space=COLOR_SPACE,
                                   spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                                   orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                                   cell_per_block=CELL_PER_BLOCK,
                                   hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                                   hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', ORIENT, 'orientations', PIX_PER_CELL,
      'pixels per cell and', CELL_PER_BLOCK, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
model = LinearSVC()
# Check the training time for the SVC
t = time.time()
model.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train model...')
# Check the score of the SVC
print('Test Accuracy of model = ', round(model.score(X_test, y_test), 4))

# save
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(X_scaler, open('X_scaler.pkl', 'wb'))

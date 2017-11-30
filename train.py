import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from utils import extract_features
from parameters import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import RandomizedSearchCV
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


def data_extract_features(data):
    return extract_features(data, color_space=COLOR_SPACE,
                            spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                            orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                            cell_per_block=CELL_PER_BLOCK,
                            hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                            hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)


def split_data():
    car_features = data_extract_features(cars)
    notcar_features = data_extract_features(notcars)
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

    return X_train, X_test, y_train, y_test, X_scaler


def load_data(regenerate=False):
    dict_data = {}
    if regenerate:
        X_train, X_test, y_train, y_test, X_scaler = split_data()
        dict_data["X_train"] = X_train
        dict_data["X_test"] = X_test
        dict_data["y_train"] = y_train
        dict_data["y_test"] = y_test
        dict_data["X_scaler"] = X_scaler

        pickle.dump(dict_data, open('data.p', 'wb'))
    else:
        dict_data = pickle.load(open("data.p", 'rb'))

    return dict_data


def train_model(dict_data):
    model = SVC(kernel="rbf", C=5.0, gamma=0.01)
    t = time.time()
    model.fit(dict_data['X_train'], dict_data['y_train'])
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train model...')
    # save
    dist_pickle = {
        'model': model,
        'X_scaler': dict_data["X_scaler"]
    }
    pickle.dump(dist_pickle, open('model_rbf_5_01.p', 'wb'))

    print('Test Accuracy of model = ', round(
        dist_pickle["model"].score(dict_data['X_test'], dict_data['y_test']), 4))


def main():
    # PRINT SOME INFO
    data_info = data_look(cars, notcars)
    dict_data = load_data()

    print(data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"],
          ' and data type:', data_info["data_type"])

    print('Using:', ORIENT, 'orientations', PIX_PER_CELL,
          'pixels per cell and', CELL_PER_BLOCK, 'cells per block')
    print('Feature vector length:', len(dict_data['X_train'][0]))

    train_model(dict_data)


if __name__ == "__main__":
    main()


# Use a SVC
# parameters = {'C': np.arange(0.0, 10 + 0.0, 1).tolist(),
#               'gamma': np.arange(0.0, 0.5 + 0.0, 0.05).tolist()}

#model = SVC(kernel='linear', C=1.0, gamma=0.1)
# clf = RandomizedSearchCV(
#     estimator=model, param_distributions=parameters, n_jobs=2, verbose=9)


# print(clf.best_estimator_)
# print(clf.best_score_)
# print(clf.best_params_)
# Check the score of the SVC

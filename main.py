import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle

from utils import slide_window, search_windows, draw_boxes
from parameters import *


image = mpimg.imread('test_images/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32) / 255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=Y_START_STOP,
                       xy_window=(96, 96), xy_overlap=(0.5, 0.5))

model = pickle.load(open("model.pkl", 'rb'))
X_scaler = pickle.load(open("X_scaler.pkl", 'rb'))


hot_windows = search_windows(image, windows, model, X_scaler, color_space=COLOR_SPACE,
                             spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                             orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                             cell_per_block=CELL_PER_BLOCK,
                             hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                             hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)
plt.show()

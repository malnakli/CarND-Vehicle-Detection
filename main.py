import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
import argparse

from utils import *
from parameters import *

image = mpimg.imread('test_images/test1.jpg')


def detect(img, dist_pickle):
    draw_image = np.copy(img)
    # the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    img = img.astype(np.float32) / 255

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=Y_START_STOP,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    model = dist_pickle["model"]
    X_scaler = dist_pickle["X_scaler"]

    hot_windows = search_windows(img, windows, model, X_scaler, color_space=COLOR_SPACE,
                                 spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                                 orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                                 cell_per_block=CELL_PER_BLOCK,
                                 hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                                 hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

    window_img = draw_boxes(draw_image, hot_windows,
                            color=(0, 0, 255), thick=6)

    # out_img = find_cars(img, Y_START_STOP[0], Y_START_STOP[1], SCALE, model,
    #                   X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS)

    return window_img


def read_video(filename='project_video.mp4', saved=False):
    cap = cv2.VideoCapture(filename)

    model = pickle.load(open("model.pkl", 'rb'))
    X_scaler = pickle.load(open("X_scaler.pkl", 'rb'))
    dist_pickle = {
        'model': model,
        'X_scaler': X_scaler
    }
    # dist_pickle = pickle.load(open("model.p", 'rb'))

    if saved:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        output = "output-" + filename
        out = cv2.VideoWriter(output, fourcc, 20.0, (1280, 720))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        img = detect(frame, dist_pickle)

        if saved:
            out.write(img)

        # Our operations on the frame come here
        # Display the resulting frame
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    read_video(filename=args.fileinput, saved=args.save_video)
    # read_test_images()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fileinput',
                        type=str, help='finename of a video file')
    parser.add_argument('-s', '--save_video', action="store_true", default=False,
                        help='Either to save the output result or not')

    main(parser.parse_args())
